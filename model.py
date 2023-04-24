"""Classes for generative model being attacked."""
from abc import ABC, abstractmethod

import openai
from transformers import pipeline

_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_MAX_TOKENS = 50


class Model(ABC):
    """Base class for generative model."""

    def __init__(
        self,
        model_uri: str,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        """Initialize the model params."""
        self.model_uri = model_uri
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Generate a text from the input text."""


class OpenAIModel(Model):
    """OpenAI model wrapper."""

    def __call__(self, text: str) -> str:
        """Generate a text from the input text."""
        completion = openai.Completion.create(
            engine=self.model_uri,
            prompt=text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].text


class HuggingFaceModel(Model):
    """HuggingFace model wrapper."""

    def __init__(
        self,
        model_uri: str,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        """Initialize the model params."""
        super().__init__(model_uri, temperature=temperature, max_tokens=max_tokens)
        self.generator = pipeline(
            "text-generation", model=self.model_uri, return_full_text=False
        )

    def __call__(self, text: str) -> str:
        """Generate a text from the input text."""
        num_input_tokens = len(self.generator.tokenizer(text)["input_ids"])
        return self.generator(
            text,
            do_sample=bool(self.temperature),
            temperature=self.temperature,
            max_length=self.max_tokens + num_input_tokens,
        )[0]["generated_text"]
