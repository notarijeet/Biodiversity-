from pathlib import Path

import pandas as pd
import pytest

from config import settings
from data.loader import group_aware_split, load_wlt_osn


def test_load_wlt_osn_file_not_found(tmp_path: Path):
    non_existent = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_wlt_osn(non_existent)


def test_load_wlt_osn_missing_columns(tmp_path: Path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(bad_csv, index=False)
    with pytest.raises((ValueError, AssertionError)):
        load_wlt_osn(bad_csv)


def test_load_wlt_osn_real_data():
    csv_path = settings.data_dir / "raw" / "share_data.csv"
    if not csv_path.exists():
        pytest.skip(f"Data file not found at {csv_path}")

    df = load_wlt_osn(csv_path)

    # Required columns
    assert set(df.columns) == {"text", "label", "source_group"}

    # Assert source is not leaking
    assert "source" not in df.columns

    # Check deduplication occurred
    assert len(df) <= 8676

    # Check positive label presence (~255 positives)
    positives = df["label"].sum()
    assert 200 <= positives <= 300, f"Unexpected positive count: {positives}"


def test_group_aware_split():
    csv_path = settings.data_dir / "raw" / "share_data.csv"
    if not csv_path.exists():
        # Fallback to synthetic grouped dataset
        df = pd.DataFrame(
            {
                "text": [f"text_{i}" for i in range(100)],
                "label": [1 if i % 10 == 0 else 0 for i in range(100)],
                "source_group": [f"grp_{i % 5}" for i in range(100)],
            }
        )
    else:
        df = load_wlt_osn(csv_path)

    train_df, test_df = group_aware_split(df, test_size=0.2, seed=42)

    train_groups = set(train_df["source_group"].unique())
    test_groups = set(test_df["source_group"].unique())

    # ZERO overlap between groups
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping groups: {overlap}"
    assert len(train_df) > 0
    assert len(test_df) > 0
