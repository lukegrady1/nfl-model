"""Step 2.4: trained logistic models with NumPy-only inference, implementing the DecisionModel interface."""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

from nflsim.config import MODELS_DIR, settings
from nflsim.models import buckets as B
from nflsim.models.features import features, situation_from_state

# Lower edge of each yardage class: a class is infeasible when its lower edge exceeds the room to the goal line.
YARD_LOWER_EDGE = {
    "air_yards": lambda c: [-99, 0, 5, 10, 15, 20, 30][int(c)],
    "yac": lambda c: [-99, 1, 2, 3, 5, 7, 10, 15, 25][int(c)],
    "rush_yards": lambda c: -99 if int(c) <= -6 else int(c),
}


def feasible_mask(name: str, classes: list, room: int) -> np.ndarray | None:
    """Mask of classes whose yardage can fit in `room` yards (None when the model has no yardage classes)."""
    edge = YARD_LOWER_EDGE.get(name)
    if edge is None:
        return None
    return np.array([edge(c) <= room for c in classes], dtype=bool)


def room_for(name: str, yardline_100: int, air_yards: float | None) -> int:
    if name == "yac":
        return max(int(yardline_100) - int(air_yards or 0), 0)
    return int(yardline_100)


def home_direction(name: str, classes: list) -> np.ndarray | None:
    """Phase 3 step 3.3: direction of the home logit offset per class. Home offense gets +h/2 along it, away -h/2."""
    if name == "throw_outcome":
        return np.array([1.0 if c == "complete" else 0.0 for c in classes])
    if name == "rush_yards":
        vals = {16: 20.0, 25: 32.0, 40: 45.0, -6: -8.0}
        return np.array([max(min((vals.get(int(c), float(c)) - 4.0) / 10.0, 1.0), -1.0) for c in classes])
    return None


def apply_mask(P: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Zero infeasible classes and renormalize (the engine would otherwise cap them, double counting the goal line)."""
    if mask is None or mask.all():
        return P
    Q = P * mask
    tot = Q.sum(axis=-1, keepdims=True)
    return np.where(tot > 0, Q / np.maximum(tot, 1e-300), P)


class TrainedModel:
    def __init__(self, spec: dict):
        self.name = spec["name"]
        self.classes = [int(c) if spec["class_type"] == "int" else c for c in spec["classes"]]
        self.feature_names = spec["feature_names"]
        self.mean = np.asarray(spec["means"], dtype=float)
        self.std = np.asarray(spec["stds"], dtype=float)
        self.W = np.asarray(spec["coef"], dtype=float)          # (K, F) for multinomial, (1, F) for binary
        self.b = np.asarray(spec["intercept"], dtype=float)
        self.binary = len(self.classes) == 2 and self.W.shape[0] == 1
        self.team = bool(spec.get("team_features", False))
        self.pf_pos = np.array([i for i, n in enumerate(self.feature_names) if n.startswith("p_")], dtype=int)
        self.home_dir = home_direction(self.name, self.classes)
        self.home_offset = float(settings().get("home_logit_offset", {}).get(self.name, 0.0)) if self.home_dir is not None else 0.0
        self.within: dict = {}
        for cls, w in spec.get("within", {}).items():
            key = int(cls) if spec["class_type"] == "int" else cls
            cum = np.cumsum(w["probs"]).tolist()
            cum[-1] = 1.0 + 1e-12
            self.within[key] = (w["values"], cum)
        self.meta = {k: spec[k] for k in ("C", "training_seasons", "validation_season", "val_log_loss", "trained_at")
                     if k in spec}

    # ---- inference --------------------------------------------------------------------------------------------
    def proba_from_features(self, x: np.ndarray, logit_shift: np.ndarray | None = None) -> np.ndarray:
        z = (x - self.mean) / self.std
        logits = z @ self.W.T + self.b
        if logit_shift is not None:
            logits = logits + logit_shift
        if self.binary:
            p1 = 1.0 / (1.0 + np.exp(-logits))
            return np.concatenate([1.0 - p1, p1], axis=-1)
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=-1, keepdims=True)

    def proba(self, state, extra: dict) -> np.ndarray:
        x = features(self.name, situation_from_state(state, extra, self.name), self.team)
        if len(self.pf_pos) and extra.get("player") is None:
            x[self.pf_pos] = self.mean[self.pf_pos]     # no player tracking: league-average player effects
        shift = None
        if self.home_offset and not state.is_neutral:
            off = extra.get("offense") or state.possession
            shift = self.home_dir * (0.5 * self.home_offset if off == state.home else -0.5 * self.home_offset)
        p = self.proba_from_features(x, shift)
        if self.name in YARD_LOWER_EDGE:
            p = apply_mask(p, feasible_mask(self.name, self.classes, room_for(self.name, state.yardline_100, extra.get("air_yards"))))
        return p

    def sample(self, rng: np.random.Generator, state, extra: dict):
        p = self.proba(state, extra)
        cum = np.cumsum(p)
        i = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        return self.classes[min(i, len(self.classes) - 1)]

    def sample_within(self, rng: np.random.Generator, cls) -> int:
        vals, cum = self.within[cls]
        i = bisect.bisect_right(cum, rng.random())
        return int(vals[min(i, len(vals) - 1)])


def model_path(name: str, directory: Path | None = None) -> Path:
    return (directory or MODELS_DIR) / f"{name}.json"


def load_trained(name: str, directory: Path | None = None) -> TrainedModel:
    with open(model_path(name, directory)) as f:
        return TrainedModel(json.load(f))


def available_trained(directory: Path | None = None) -> list[str]:
    d = directory or MODELS_DIR
    return sorted(p.stem for p in d.glob("*.json") if p.stem != "report")
