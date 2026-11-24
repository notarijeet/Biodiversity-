import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_wlt_osn(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate the Mou et al. WLT-OSN dataset.
    Returns a cleaned DataFrame with columns: text, label, source_group.
    NEVER includes 'source' column as a feature.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Determine text column
    if "tweet_text_cleaned" in df.columns:
        text_col = "tweet_text_cleaned"
    elif "cleaned_text" in df.columns:
        text_col = "cleaned_text"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError(
            "Missing text column in dataset (expected 'tweet_text_cleaned' or 'cleaned_text')"
        )

    required = {text_col, "label", "source"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    # Drop null values
    df = df.dropna(subset=[text_col, "label", "source"]).copy()

    # Deduplicate by text hash
    df["_text_hash"] = df[text_col].apply(lambda x: hashlib.md5(str(x).encode("utf-8")).hexdigest())
    df = df.drop_duplicates(subset="_text_hash").drop(columns="_text_hash")

    # Keep only needed columns (explicitly exclude 'source' from features)
    df = df[[text_col, "label", "source"]].rename(
        columns={text_col: "text", "source": "source_group"}
    )

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    # Report class balance
    print(
        f"Loaded {len(df)} rows | Positives: {df['label'].sum()} ({df['label'].mean() * 100:.1f}%)"
    )
    print(f"Source groups: {len(df['source_group'].unique())} unique groups")

    return df.reset_index(drop=True)


def group_aware_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by source_group to prevent data leakage.
    Training set and test set will not share any source groups.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["source_group"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


# Alias for compatibility
get_group_splits = group_aware_split
