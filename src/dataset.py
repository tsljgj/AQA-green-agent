"""Dataset loader for AQA benchmark questions."""
import json
import random
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


# Map expansion_level to difficulty names
LEVEL_TO_DIFFICULTY = {
    0: "easy",
    1: "medium",
    2: "hard",
    3: "expert"
}

# Ordered difficulty levels (for display purposes)
DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]


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
        """Load dataset from directory containing tier folders.

        Args:
            dataset_path: Path to the dataset directory (containing tier1/, tier2/ folders)
        """
        self.dataset_path = Path(dataset_path)
        self.questions: list[Question] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load questions from dataset directory.

        Supports two formats:
        1. Flat structure: JSON files directly in dataset_path
        2. Tiered structure: JSON files in tier1/, tier2/ subdirectories
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Check for flat structure first (files directly in directory)
        json_files = list(self.dataset_path.glob("*.json"))

        if json_files:
            # Flat structure - load directly
            for json_file in json_files:
                self._load_question_file(json_file)
        else:
            # Tiered structure - load from tier1 and tier2 directories
            for tier_dir in ["tier1", "tier2"]:
                tier_path = self.dataset_path / tier_dir
                if not tier_path.exists():
                    continue

                for json_file in tier_path.glob("*.json"):
                    self._load_question_file(json_file, tier=tier_dir)

    def _load_question_file(self, json_file: Path, tier: str | None = None) -> None:
        """Load a single question from a JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Map fields from new format to expected format
        expansion_level = data.get("expansion_level", 0)
        difficulty = LEVEL_TO_DIFFICULTY.get(expansion_level, "medium")

        question = Question(
            qid=json_file.stem,
            difficulty=difficulty,
            question=data["question"],
            reference_answer=data["expected_answer"],
            domain=data.get("subject"),
            metadata={
                "tier": tier,
                "expansion_level": expansion_level,
                "source_file": data.get("source_file"),
                "assessment": data.get("assessment"),
            }
        )
        self.questions.append(question)

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
