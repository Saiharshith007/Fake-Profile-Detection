"""Load the MIB dataset of X (Twitter) accounts: genuine (E13) vs. fake (INT)."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Profile activity counts only. `lang` and a name-derived gender feature are
# not used: neither improves accuracy, and `lang` mostly encodes which dataset a
# row came from (genuine accounts are Italian, fake ones English).
FEATURES = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
]


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Return features X and labels y, where 1 = fake and 0 = genuine."""
    # Label each file before concatenating so labels can't drift from their rows.
    genuine = pd.read_csv(data_dir / "genuine.csv", usecols=FEATURES).assign(is_fake=0)
    fake = pd.read_csv(data_dir / "fake.csv", usecols=FEATURES).assign(is_fake=1)
    df = pd.concat([genuine, fake], ignore_index=True)
    return df[FEATURES], df["is_fake"]
