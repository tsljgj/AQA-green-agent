"""LLM-as-judge evaluator for answer scoring."""
import json
import os
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """Result of evaluating a single answer."""
    correct: bool
    score: float  # 0.0 to 1.0
    reasoning: str


class LLMEvaluator:
    """Evaluates agent answers using an LLM as judge."""

    EVAL_PROMPT_TEMPLATE = """You are evaluating an agent's answer to a question.

Question: {question}
Reference Answer: {reference_answer}
Agent's Answer: {agent_answer}

Determine if the agent's answer is semantically equivalent to the reference answer.

Consider:
- Exact matches are correct
- Semantically equivalent phrasings are correct (e.g., "Paris" vs "Paris, France")
- Synonyms with the same meaning are correct
- Minor formatting differences are acceptable (e.g., "H2O" vs "H₂O")
- Different information or wrong answers are incorrect
- Partial answers may receive partial credit (0.0-1.0 score)

Respond ONLY with valid JSON in this exact format:
{{
  "correct": true or false,
  "score": a number between 0.0 and 1.0,
  "reasoning": "brief explanation in one sentence"
}}"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize the LLM evaluator.

        Args:
            model: OpenAI model to use for evaluation
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def evaluate(
        self,
        question: str,
        reference_answer: str,
        agent_answer: str
    ) -> EvaluationResult:
        """Evaluate an agent's answer using LLM-as-judge.

        Args:
            question: The question that was asked
            reference_answer: The expected/reference answer
            agent_answer: The agent's actual answer

        Returns:
            EvaluationResult with correctness, score, and reasoning
        """
        prompt = self.EVAL_PROMPT_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            agent_answer=agent_answer
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise answer evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic evaluation
                max_tokens=200
            )

            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("Empty response from LLM")

            # Parse JSON response
            result_data = json.loads(result_text.strip())

            return EvaluationResult(
                correct=result_data["correct"],
                score=float(result_data["score"]),
                reasoning=result_data["reasoning"]
            )

        except json.JSONDecodeError as e:
            # Fallback: if LLM doesn't return valid JSON, do string matching
            return self._fallback_evaluation(reference_answer, agent_answer)
        except Exception as e:
            print(f"Evaluation error: {e}")
            # Fallback to simple string matching
            return self._fallback_evaluation(reference_answer, agent_answer)

    def _fallback_evaluation(self, reference: str, answer: str) -> EvaluationResult:
        """Simple fallback evaluation if LLM fails.

        Args:
            reference: Reference answer
            answer: Agent's answer

        Returns:
            EvaluationResult based on exact string matching
        """
        # Normalize strings for comparison
        ref_normalized = reference.strip().lower()
        ans_normalized = answer.strip().lower()

        # Check for exact match or containment
        if ref_normalized == ans_normalized:
            return EvaluationResult(
                correct=True,
                score=1.0,
                reasoning="Exact match (fallback evaluation)"
            )
        elif ref_normalized in ans_normalized or ans_normalized in ref_normalized:
            return EvaluationResult(
                correct=True,
                score=0.8,
                reasoning="Partial match (fallback evaluation)"
            )
        else:
            return EvaluationResult(
                correct=False,
                score=0.0,
                reasoning="No match (fallback evaluation)"
            )
