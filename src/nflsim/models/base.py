"""Step 1.2: the decision-model interface shared by lookup tables (Phase 1) and trained models (Phase 2)."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DecisionModel(Protocol):
    name: str
    classes: list

    def proba(self, state: Any, extra: dict) -> np.ndarray:  # sums to 1
        ...

    def sample(self, rng: np.random.Generator, state: Any, extra: dict) -> object:
        ...


class FixedModel:
    """Test stub: always returns one class (or a fixed probability vector)."""

    def __init__(self, name: str, classes: list, fixed: object = None, probs: list[float] | None = None):
        self.name = name
        self.classes = list(classes)
        self.fixed = fixed
        self._probs = np.array(probs, dtype=float) if probs is not None else None

    def proba(self, state, extra):
        if self._probs is not None:
            return self._probs
        p = np.zeros(len(self.classes))
        p[self.classes.index(self.fixed)] = 1.0
        return p

    def sample(self, rng, state, extra):
        if self._probs is None:
            return self.fixed
        return self.classes[int(rng.choice(len(self.classes), p=self._probs))]
