"""Dataset loader for AQA benchmark questions."""
import json
import random
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class Question(BaseModel):
    """A single QA question."""
    qid: str
    difficulty: str
    question: str
    reference_answer: str
    domain: Optional[str] = None
    metadata: Optional[dict] = None


class QADataset:
    """Loads and samples questions from the AQA dataset."""

    def __init__(self, dataset_path: str | Path):
        """Load dataset from JSONL file.

        Args:
            dataset_path: Path to the JSONL dataset file
        """
        self.dataset_path = Path(dataset_path)
        self.questions: list[Question] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load questions from JSONL file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.questions.append(Question(**data))

    def sample_questions(
        self,
        num_questions: int,
        difficulties: Optional[list[str]] = None,
        domains: Optional[list[str]] = None,
        seed: Optional[int] = None
    ) -> list[Question]:
        """Sample questions based on filters.

        Args:
            num_questions: Number of questions to sample
            difficulties: Filter by difficulty levels (e.g., ["easy", "medium"])
            domains: Filter by domains (e.g., ["geography", "math"])
            seed: Random seed for reproducibility

        Returns:
            List of sampled Question objects
        """
        # Filter questions
        filtered = self.questions

        if difficulties:
            filtered = [q for q in filtered if q.difficulty in difficulties]

        if domains:
            filtered = [q for q in filtered if q.domain in domains]

        if not filtered:
            raise ValueError("No questions match the specified filters")

        # Sample questions
        if seed is not None:
            random.seed(seed)

        # If requesting more questions than available, sample with replacement
        if num_questions > len(filtered):
            return random.choices(filtered, k=num_questions)
        else:
            return random.sample(filtered, k=num_questions)

    def get_question_by_id(self, qid: str) -> Optional[Question]:
        """Get a specific question by ID.

        Args:
            qid: Question ID

        Returns:
            Question object if found, None otherwise
        """
        for q in self.questions:
            if q.qid == qid:
                return q
        return None

    def get_difficulty_distribution(self) -> dict[str, int]:
        """Get count of questions by difficulty level.

        Returns:
            Dictionary mapping difficulty -> count
        """
        distribution: dict[str, int] = {}
        for q in self.questions:
            distribution[q.difficulty] = distribution.get(q.difficulty, 0) + 1
        return distribution

    def __len__(self) -> int:
        """Return total number of questions."""
        return len(self.questions)

    def __repr__(self) -> str:
        """String representation of dataset."""
        dist = self.get_difficulty_distribution()
        return f"QADataset({len(self)} questions, difficulties: {dist})"
