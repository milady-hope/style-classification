import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from src.config import DATA_PATH, SEED, TOKEN_RE_PAT, SENT_RE_PAT

_tok = re.compile(TOKEN_RE_PAT)
_sent = re.compile(SENT_RE_PAT)
_url = re.compile(r"https?://\S+|www\.\S+")
_email = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b")
_sp = re.compile(r"\s+")


def tokenize(text: str):
    return _tok.findall(text.lower())


def split_sents(text: str):
    ss = [s.strip() for s in _sent.split(text) if s.strip()]
    return ss if ss else [text.strip()]


def clean(text: str) -> str:
    s = text.replace("\u200b", "")
    s = _url.sub(" ", s)
    s = _email.sub(" ", s)
    s = s.replace("\t", " ").replace("\xa0", " ")
    return _sp.sub(" ", s).strip()


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, header=None)
    df.columns = ["text", "label", "pair_id"]
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df["pair_id"] = df["pair_id"].astype(int)
    df["text_clean"] = df["text"].map(clean)
    return df


def train_test_by_pairs(df, test_size=0.2, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df["text_clean"], df["label"], groups=df["pair_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def group_kfold(n_splits=5):
    return GroupKFold(n_splits=n_splits)


def build_pairs_df(df):
    src = df[df["label"] == 0][["pair_id", "text"]].rename(columns={"text": "source"})
    tgt = df[df["label"] == 1][["pair_id", "text"]].rename(columns={"text": "target"})
    return src.merge(tgt, on="pair_id", how="inner").dropna().reset_index(drop=True)
