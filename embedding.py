"""Class for computing token embeddings."""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Optional, Sequence, Dict

import torch
from transformers import AutoTokenizer, AutoModel


class Embedding(ABC):
    """Class for computing token embeddings."""

    def __init__(
        self,
        forbidden_tokens: Optional[Sequence[str]] = None,
        allowed_tokens: Optional[Sequence[str]] = None,
    ):
        """Initialize the embedding params."""
        self.all_tokens = self._get_all_tokens(forbidden_tokens, allowed_tokens)
        self.all_token_ids = [self.vocab_dict[token] for token in self.all_tokens]

    def _get_all_tokens(
        self,
        forbidden_tokens: Optional[Sequence[str]] = None,
        allowed_tokens: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Get all token tokens in constrained vocab."""
        if forbidden_tokens is None and allowed_tokens is None:
            return list(self.vocab_dict.keys())
        vocab = set(self.vocab_dict.keys())
        allowed_tokens = set(allowed_tokens or [])
        forbidden_tokens = set(forbidden_tokens or [])
        all_tokens = vocab.intersection(allowed_tokens) - forbidden_tokens
        if len(all_tokens) == 0:
            raise ValueError("Candidate token set is empty.")
        return list(all_tokens)

    @abstractmethod
    def get_embeddings(self, text: List[str]) -> torch.Tensor:
        """Compute the matrix of embeddings for a given text."""

    @abstractmethod
    def project_embeddings(self, X_batch: torch.Tensor) -> str:
        """Project the embedding matrix onto the token space and decode to a string for each prompt in the batch."""

    @property
    @abstractmethod
    def vocab_dict(self) -> Dict[str, int]:
        """Get the vocab mapping of tokens to ids for the embedding model."""

    @cached_property
    @abstractmethod
    def all_token_embeddings(self) -> torch.Tensor:
        """Get the embeddings for all tokens in the filtered vocabulary."""

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of tokens to a string."""


class HuggingFaceEmbedding(Embedding):
    """Class for computing token embeddings using HuggingFace models."""

    def __init__(
        self,
        model_uri: str,
        forbidden_tokens: Optional[Sequence[str]] = None,
        allowed_tokens: Optional[Sequence[str]] = None,
    ):
        """Initialize the HuggingFaceEmbedding class."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_uri)
        model = AutoModel.from_pretrained(model_uri)
        self.embedding_module = model.get_input_embeddings()
        super().__init__(forbidden_tokens, allowed_tokens)

    @property
    def vocab_dict(self) -> Dict[str, int]:
        """Get the vocab mapping of tokens to ids for the embedding model."""
        return self.tokenizer.get_vocab()

    @cached_property
    def all_token_embeddings(self) -> torch.Tensor:
        """Get the embeddings for all tokens in the filtered vocabulary."""
        return self.embedding_module(torch.tensor(self.all_token_ids))

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Compute the matrix of embeddings for a given text."""
        input_ids = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        return self.embedding_module(input_ids).detach()

    def project_embeddings(self, X_batch: torch.Tensor) -> List[str]:
        """Project the embedding matrix onto the token space and decode to a string for each prompt in the batch."""
        if X_batch.ndim == 2:
            X_batch = X_batch.reshape(
                X_batch.shape[0], -1, self.embedding_module.embedding_dim
            )
        projected_texts = []
        for X in X_batch:
            distances = torch.norm(self.all_token_embeddings.unsqueeze(1) - X, dim=2)
            closest_tokens = torch.tensor(
                [self.all_token_ids[token] for token in torch.argmin(distances, axis=0)]
            )
            projected_texts.append(
                self.tokenizer.decode(closest_tokens, skip_special_tokens=True)
            )
        return projected_texts

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of tokens to a string."""
        return self.tokenizer.convert_tokens_to_string(tokens)
