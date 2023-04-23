"""Classes for objective functions."""
from abc import ABC, abstractmethod
from typing import List, Union, cast

import evaluate
from nltk import sent_tokenize
import pandas as pd

base_toxicity_metric = evaluate.load("toxicity")


# TODO: consider having the objective wrap the model as well

class Objective(ABC):
    """Class representing an objective function to be optimized."""

    @abstractmethod
    def __call__(self, texts: List[str]) -> float:
        """Compute the objective value for a given text batch."""


class ToxicityObjective(Objective):
    """Toxicity score objective function."""

    def __call__(self, texts: List[str]) -> float:
        """Compute the objective value for a given text batch."""
        return toxicity_classifier(texts).mean()


def toxicity_classifier(texts: Union[List[str], pd.Series]) -> pd.Series:
    """Return the toxicity scores for each text."""
    # Empirically, splitting each text into separate sentences and aggregating over
    # their scores seems to work better than classifying the full text.
    texts_series = pd.Series(texts) if isinstance(texts, list) else texts
    sentences = cast(pd.Series, texts_series.apply(sent_tokenize)).explode()
    sentence_scores = pd.Series(
        base_toxicity_metric.compute(predictions=sentences.to_list())["toxicity"],
        index=sentences.index,
    )
    # TODO: maybe should be max instead of mean?
    return sentence_scores.groupby(sentence_scores.index).mean()
