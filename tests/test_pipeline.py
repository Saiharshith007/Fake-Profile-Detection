import pandas as pd
from sklearn.model_selection import train_test_split

from fake_profiles.data import FEATURES, load_dataset
from fake_profiles.train import build_models, evaluate


def test_labels_match_source_file(tmp_path):
    # Each row must keep the label of the file it came from, whatever the concat order.
    pd.DataFrame({f: [0, 0] for f in FEATURES}).to_csv(tmp_path / "genuine.csv", index=False)
    pd.DataFrame({f: [9, 9, 9] for f in FEATURES}).to_csv(tmp_path / "fake.csv", index=False)

    X, y = load_dataset(tmp_path)

    assert y.tolist() == [0, 0, 1, 1, 1]
    assert (X.loc[y == 1, "statuses_count"] == 9).all()


def test_models_beat_baseline_on_real_data():
    X, y = load_dataset()
    split = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    results = evaluate(build_models(), *split)

    assert (results["accuracy"] > 0.95).all(), results
