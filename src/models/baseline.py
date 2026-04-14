from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import SEED


def build_baseline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="liblinear", max_iter=2000, random_state=SEED
        )),
    ])
