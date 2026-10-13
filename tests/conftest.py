import pytest

from nflsim.config import PROCESSED_DIR


def has_processed() -> bool:
    return (PROCESSED_DIR / "plays.parquet").exists() and (PROCESSED_DIR / "schedules.parquet").exists()


needs_data = pytest.mark.skipif(not has_processed(), reason="processed data not built (run nflsim ingest && nflsim clean)")
