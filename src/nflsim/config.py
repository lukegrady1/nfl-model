"""Project paths and settings loader. All tunable numbers live in config/settings.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
ARTIFACTS_DIR = ROOT / "artifacts"
# Backtest mode (Phase 7): a walk-forward season runs in a subprocess with these overrides so every model, table and
# derived (ratings / player) table it touches was built from seasons before the test season.
import os as _os
MODELS_DIR = Path(_os.environ.get("NFLSIM_MODELS_DIR", ARTIFACTS_DIR / "models"))
TABLES_DIR = Path(_os.environ.get("NFLSIM_TABLES_DIR", ARTIFACTS_DIR / "tables"))
DERIVED_DIR = Path(_os.environ.get("NFLSIM_DERIVED_DIR", PROCESSED_DIR))   # team_ratings / player tables


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    with open(CONFIG_DIR / "settings.yaml") as f:
        s = yaml.safe_load(f)
    # backtest overrides (see MODELS_DIR above)
    if _os.environ.get("NFLSIM_TRAINING_SEASONS"):
        s["training_seasons"] = parse_seasons(_os.environ["NFLSIM_TRAINING_SEASONS"])
    if _os.environ.get("NFLSIM_TABLE_SEASONS"):
        s["table_seasons"] = parse_seasons(_os.environ["NFLSIM_TABLE_SEASONS"])
    if _os.environ.get("NFLSIM_KICKOFF_SEASONS"):
        s["kickoff_seasons"] = parse_seasons(_os.environ["NFLSIM_KICKOFF_SEASONS"])
    if _os.environ.get("NFLSIM_INPUT_SCALE"):
        s["team_strength"]["input_scale"] = float(_os.environ["NFLSIM_INPUT_SCALE"])
    return s


def team_aliases() -> dict[str, str]:
    with open(CONFIG_DIR / "team_aliases.yaml") as f:
        raw = yaml.safe_load(f)
    # YAML 1.1 turns a bare NO / YES / ON / OFF into a boolean; the Saints are "NO"
    return {k: ("NO" if v is False else str(v)) for k, v in raw.items()}


def parse_seasons(spec: str) -> list[int]:
    """'2018-2026' -> [2018..2026]; '2023,2025' -> [2023, 2025]; '2024' -> [2024]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))
