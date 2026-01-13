import time
from pathlib import Path
from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from dataset import QADataset, Question
from evaluator import LLMEvaluator, EvaluationResult


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class QuestionResult(BaseModel):
    """Result for a single question."""
    qid: str
    difficulty: str
    question: str
    reference_answer: str
    agent_answer: str
    correct: bool
    score: float
    latency_ms: int
    evaluation_reasoning: str
    domain: str | None = None


class AggregateResults(BaseModel):
    """Aggregate metrics across all questions."""
    total_tasks: int
    correct: int
    pass_rate: float
    avg_score: float
    total_time_ms: int
    avg_latency_ms: int
    by_difficulty: dict[str, dict[str, Any]]


class AssessmentResult(BaseModel):
    """Complete assessment result."""
    assessment_id: str
    config: dict[str, Any]
    items: list[QuestionResult]
    aggregate: AggregateResults
    metadata: dict[str, Any]


class Agent:
    """AQA Green Agent - QA benchmark evaluator."""

    # Required participant roles
    required_roles: list[str] = ["agent"]  # The purple agent being tested

    # Required config keys
    required_config_keys: list[str] = ["num_tasks"]

    def __init__(self):
        self.messenger = Messenger()

        # Load dataset
        dataset_path = Path(__file__).parent.parent / "data" / "sample_questions.jsonl"
        self.dataset = QADataset(dataset_path)

        # Initialize evaluator (will use OPENAI_API_KEY from environment)
        self.evaluator = LLMEvaluator()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the assessment request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Validate num_tasks
        num_tasks = request.config.get("num_tasks", 0)
        if not isinstance(num_tasks, int) or num_tasks <= 0:
            return False, "num_tasks must be a positive integer"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Run the AQA benchmark assessment.

        Args:
            message: The incoming assessment request message
            updater: TaskUpdater for reporting progress and results
        """
        input_text = get_message_text(message)

        # Parse and validate request
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Extract config parameters
        config = request.config
        num_tasks = config["num_tasks"]
        difficulties = config.get("difficulty", ["easy", "medium", "hard"])
        domains = config.get("domains", None)
        seed = config.get("seed", None)
        timeout_per_question = config.get("timeout_per_question", 30)
        evaluator_model = config.get("evaluator_model", "gpt-4o-mini")

        # Get purple agent URL
        agent_url = str(request.participants["agent"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Loading {num_tasks} questions from AQA dataset...")
        )

        # Sample questions
        try:
            questions = self.dataset.sample_questions(
                num_questions=num_tasks,
                difficulties=difficulties,
                domains=domains,
                seed=seed
            )
        except ValueError as e:
            await updater.reject(new_agent_text_message(f"Dataset error: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Sampled {len(questions)} questions. Starting assessment...")
        )

        # Run assessment
        results: list[QuestionResult] = []
        total_time_ms = 0

        for i, question in enumerate(questions, 1):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Question {i}/{num_tasks}: {question.question[:50]}...")
            )

            # Ask purple agent
            start_time = time.time()
            try:
                agent_answer = await self.messenger.talk_to_agent(
                    message=question.question,
                    url=agent_url
                )
            except Exception as e:
                agent_answer = f"[Error: {str(e)}]"

            latency_ms = int((time.time() - start_time) * 1000)
            total_time_ms += latency_ms

            # Evaluate answer
            try:
                eval_result: EvaluationResult = await self.evaluator.evaluate(
                    question=question.question,
                    reference_answer=question.reference_answer,
                    agent_answer=agent_answer
                )
            except Exception as e:
                # Fallback evaluation
                eval_result = EvaluationResult(
                    correct=False,
                    score=0.0,
                    reasoning=f"Evaluation failed: {str(e)}"
                )

            # Record result
            results.append(QuestionResult(
                qid=question.qid,
                difficulty=question.difficulty,
                question=question.question,
                reference_answer=question.reference_answer,
                agent_answer=agent_answer,
                correct=eval_result.correct,
                score=eval_result.score,
                latency_ms=latency_ms,
                evaluation_reasoning=eval_result.reasoning,
                domain=question.domain
            ))

        # Calculate aggregate metrics
        correct_count = sum(1 for r in results if r.correct)
        pass_rate = correct_count / len(results) if results else 0.0
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        avg_latency_ms = total_time_ms // len(results) if results else 0

        # Calculate by-difficulty breakdown
        by_difficulty: dict[str, dict[str, Any]] = {}
        for difficulty in set(r.difficulty for r in results):
            diff_results = [r for r in results if r.difficulty == difficulty]
            diff_correct = sum(1 for r in diff_results if r.correct)
            diff_total = len(diff_results)
            by_difficulty[difficulty] = {
                "correct": diff_correct,
                "total": diff_total,
                "pass_rate": diff_correct / diff_total if diff_total > 0 else 0.0,
                "avg_score": sum(r.score for r in diff_results) / diff_total if diff_total > 0 else 0.0
            }

        aggregate = AggregateResults(
            total_tasks=len(results),
            correct=correct_count,
            pass_rate=pass_rate,
            avg_score=avg_score,
            total_time_ms=total_time_ms,
            avg_latency_ms=avg_latency_ms,
            by_difficulty=by_difficulty
        )

        # Create final assessment result
        assessment_result = AssessmentResult(
            assessment_id=f"aqa-{int(time.time())}",
            config=config,
            items=results,
            aggregate=aggregate,
            metadata={
                "purple_agent_url": agent_url,
                "dataset_size": len(self.dataset),
                "evaluator_model": evaluator_model,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )

        # Report results
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Assessment complete! Pass rate: {pass_rate:.1%} ({correct_count}/{len(results)})"
            )
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(
                    text=f"AQA Benchmark Results\n\n"
                         f"Pass Rate: {pass_rate:.1%} ({correct_count}/{len(results)})\n"
                         f"Average Score: {avg_score:.3f}\n"
                         f"Total Time: {total_time_ms}ms\n"
                         f"Average Latency: {avg_latency_ms}ms\n\n"
                         f"By Difficulty:\n" +
                         "\n".join(f"  {diff}: {stats['correct']}/{stats['total']} "
                                   f"({stats['pass_rate']:.1%})"
                                   for diff, stats in by_difficulty.items())
                )),
                Part(root=DataPart(data=assessment_result.model_dump()))
            ],
            name="AQA Assessment Result",
        )
