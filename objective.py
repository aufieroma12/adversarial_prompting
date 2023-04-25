"""Classes for objective functions."""
from abc import ABC, abstractmethod
from typing import List, cast

import evaluate
from nltk import sent_tokenize
import pandas as pd


# TODO: consider having the objective wrap the model as well

class Objective(ABC):
    """Class representing an objective function to be optimized."""

    @abstractmethod
    def __call__(self, texts: List[str]) -> float:
        """Compute the objective value for a given text batch."""


class ToxicityObjective(Objective):
    """Toxicity score objective function."""

    def __init__(self):
        """Initialize the underlying evaluation model."""
        self.base_toxicity_metric = evaluate.load("toxicity")

    def __call__(self, texts: List[str]) -> float:
        """Compute the objective value for a given text batch."""
        # Empirically, splitting each text into separate sentences and aggregating over
        # their scores seems to work better than classifying the full text.
        texts_series = pd.Series(texts)
        sentences = cast(pd.Series, texts_series.apply(_safe_sent_tokenize)).explode()
        sentence_scores = pd.Series(
            self.base_toxicity_metric.compute(
                predictions=sentences.to_list()
            )["toxicity"],
            index=sentences.index,
        )
        return sentence_scores.groupby(sentence_scores.index).max().mean()


def _safe_sent_tokenize(text: str) -> List[str]:
    """Return a list of sentences for a given text."""
    sentences = sent_tokenize(text)
    return sentences or [""]
