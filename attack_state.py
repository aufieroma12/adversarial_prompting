"""Class for tracking the attack state."""
from dataclasses import dataclass
from typing import List, Optional

import torch

from embedding import Embedding
from model import Model
from objective import Objective


@dataclass
class Input:
    """Class for storing the additional details for an attacked input."""

    prompt_template: str = "{}"
    static_instruction: str = ""

    def get_full_text(self, variable_text: str) -> str:
        """Format all text components into a single string."""
        return self.prompt_template.format(variable_text + self.static_instruction)


class AttackState:
    """Class for tracking the attack state."""

    def __init__(
        self,
        inputs: List[Input],
        embedding_module: Embedding,
        model: Model,
        objective: Objective,
        max_iters: int,
        max_consecutive_unsuccessful_iters: int,
        max_queries: int,
    ):
        """Initialize attack state params."""
        self.inputs = inputs
        self.embedding_module = embedding_module
        self.model = model
        self.objective = objective
        self.max_iters = max_iters
        self.max_consecutive_unsuccessful_iters = max_consecutive_unsuccessful_iters
        self.max_queries = max_queries

        self.consecutive_unsuccessful_iters = 0
        self.num_iters = 0
        self._num_queries = 0
        self._variable_text: Optional[str] = None
        self._score: Optional[float] = None

    def get_full_texts(self) -> List[str]:
        """Return the full text for each input."""
        return [input_.get_full_text(self.variable_text) for input_ in self.inputs]

    @property
    def num_queries(self) -> int:
        """Return the current number of queries."""
        return self._num_queries

    @property
    def score(self) -> float:
        """Return the current score."""
        if self._score is None:
            raise ValueError("Score is not yet set.")
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        """Set the value of the score property."""
        self._score = score

    @property
    def variable_text(self) -> str:
        """Return the current variable text."""
        if self._variable_text is None:
            raise ValueError("Variable text is not yet set.")
        return self._variable_text

    @variable_text.setter
    def variable_text(self, variable_text: str) -> None:
        """Set the value of the variable_text property."""
        self._variable_text = variable_text

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate a batch."""
        projected_texts = self.embedding_module.project_embeddings(X)
        scores = []
        for text in projected_texts:
            batch = [input_.get_full_text(text) for input_ in self.inputs]
            outputs = [self.model(text) for text in batch]
            self._num_queries += len(batch)
            scores.append(self.objective(outputs))
        return torch.tensor(scores)

    @property
    def exhausted(self) -> bool:
        """Return whether the attack budget has been exhausted."""
        return (
            self.num_iters >= self.max_iters
            or self.consecutive_unsuccessful_iters
            >= self.max_consecutive_unsuccessful_iters
            or self.num_queries >= self.max_queries
        )
