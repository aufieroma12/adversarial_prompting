"""Class handling the end-to-end attack logic."""
from dataclasses import dataclass
from typing import List, Optional

import gpytorch
from gpytorch.mlls import PredictiveLogLikelihood
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from attack_state import AttackState, Input
from embedding import Embedding
from model import Model
from objective import Objective
from utils.bo_utils.ppgpr import GPModelDKL
from utils.bo_utils.trust_region import TrustRegionState, generate_batch, update_state

_DEFAULT_NUM_TOKENS = 5
_DEFAULT_NUM_CANDIDATES = 5
_DEFAULT_LEARNING_RATE = 0.01
_DEFAULT_NUM_EPOCHS = 2
_DEFAULT_MAX_QUERIES = 1000
_DEFAULT_MAX_ITERS = 500
_DEFAULT_MAX_CONSECUTIVE_UNSUCCESSFUL_ITERS = 10


@dataclass(frozen=True)
class AttackConfig:
    """Config containing attack params."""

    num_tokens: int = _DEFAULT_NUM_TOKENS
    num_candidates: int = _DEFAULT_NUM_CANDIDATES
    learning_rate: float = _DEFAULT_LEARNING_RATE
    num_epochs: int = _DEFAULT_NUM_EPOCHS
    max_queries: int = _DEFAULT_MAX_QUERIES
    max_iters: int = _DEFAULT_MAX_ITERS
    max_consecutive_unsuccessful_iters: int = _DEFAULT_MAX_CONSECUTIVE_UNSUCCESSFUL_ITERS


class Attack:
    """Class handling the end-to-end attack logic."""

    def __init__(
        self,
        embedding_module: Embedding,
        model: Model,
        objective: Objective,
        attack_config: Optional[AttackConfig] = None,
    ):
        """Initialize the attack attributes."""
        self.embedding_module = embedding_module
        self.model = model
        self.objective = objective
        self.config = attack_config or AttackConfig()

    def score(self, X: torch.Tensor, inputs: List[Input]) -> torch.Tensor:
        """Compute the score for a given batch of inputs."""
        projected_texts = self.embedding_module.project_embeddings(X)
        scores = []
        for text in projected_texts:
            batch = [input_.get_full_text(text) for input_ in inputs]
            outputs = [self.model(text) for text in batch]
            scores.append(self.objective(outputs))
        return torch.tensor(scores)

    def run(
        self, inputs: List[Input], initial_prompt: Optional[str] = None
    ) -> AttackState:
        """Run the attack over the provided inputs."""
        initial_prompts = (
            [initial_prompt]
            if initial_prompt is not None
            else self._get_initial_prompts(
                self.config.num_tokens, self.config.num_candidates
            )
        )
        X = self.embedding_module.get_embeddings(initial_prompts).flatten(1, 2)
        y = self.score(X, inputs)
        surrogate_model = self._initialize_surrogate_model(X)
        surrogate_model = self._update_surrogate_model(
            surrogate_model, X, y, self.config.learning_rate, self.config.num_epochs
        )
        y_best = y.max().item()
        x_best = X[y.argmax()]
        num_trust_region_restarts = 0
        dim = X.shape[1]
        trust_region = TrustRegionState(dim=dim)
        consecutive_unsuccessful_iters = 0

        for num_iters in range(1, self.config.max_iters + 1):
            X_next = generate_batch(
                trust_region, surrogate_model, x_best, y_best, self.config.num_candidates
            )
            y_next = self.score(X_next, inputs)

            X = torch.cat((X, X_next), dim=0)
            y = torch.cat((y, y_next), dim=0)

            trust_region = update_state(trust_region, y_next)
            if trust_region.restart_triggered:
                num_trust_region_restarts += 1
                trust_region = TrustRegionState(dim=dim)
            if (
                trust_region.restart_triggered
                or ((X.shape[0] < 1024) and (num_iters % 10 == 0))
            ):
                # restart gp and update on all data
                surrogate_model = self._initialize_surrogate_model(X)
                surrogate_model = self._update_surrogate_model(
                    surrogate_model,
                    X,
                    y,
                    self.config.learning_rate,
                    self.config.num_epochs,
                )
            else:
                surrogate_model = self._update_surrogate_model(
                    surrogate_model,
                    X_next,
                    y_next,
                    self.config.learning_rate,
                    self.config.num_epochs,
                )

            if y_next.max().item() > y_best:
                consecutive_unsuccessful_iters = 0
                y_best = y_next.max().item()
                x_best = X_next[y_next.argmax()]
            else:
                consecutive_unsuccessful_iters += 1

            if (
                consecutive_unsuccessful_iters > self.config.max_consecutive_unsuccessful_iters  # pylint: disable=line-too-long
                or self.model.num_queries > self.config.max_queries
            ):
                break

        best_prompt = self.embedding_module.project_embeddings(x_best.unsqueeze(0))[0]
        return AttackState(best_prompt, inputs, y_best)

    def _get_initial_prompts(self, num_tokens: int, num_candidates: int):
        """Return a randomly initialized set of candidate prompts."""
        prompts = []
        for _ in range(num_candidates):
            tokens = np.random.choice(self.embedding_module.all_tokens, size=num_tokens)
            prompts.append(self.embedding_module.convert_tokens_to_string(tokens))
        return prompts

    def _initialize_surrogate_model(self, X: torch.Tensor) -> GPModelDKL:
        """Return the initialized surrogate model."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        surrogate_model = GPModelDKL(
            X, likelihood=likelihood, hidden_dims=(256, 128, 64)
        )
        return surrogate_model.eval()

    def _update_surrogate_model(
        self,
        model: GPModelDKL,
        X: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float,
        epochs: int,
    ) -> GPModelDKL:
        model = model.train()
        mll = PredictiveLogLikelihood(model.likelihood, model, num_data=X.shape[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        batch_size = min(len(y), 128)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for (inputs, scores) in train_loader:
                optimizer.zero_grad()
                output = model(inputs)
                loss = -mll(output, scores).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        return model.eval()
