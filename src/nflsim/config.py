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
MODELS_DIR = ARTIFACTS_DIR / "models"
TABLES_DIR = ARTIFACTS_DIR / "tables"


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    with open(CONFIG_DIR / "settings.yaml") as f:
        return yaml.safe_load(f)


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
