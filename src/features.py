import numpy as np
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

from src.config import PRONOUNS, DISCOURSE
from src.preprocessing import tokenize, split_sents

STOP_RU = set(stopwords.words("russian"))

FEAT_NAMES = [
    "log_len", "pron_share", "stop_share",
    "avg_sent_len", "std_sent_len", "ttr",
    "digit_share", "latin_share",
    "quest_share", "excl_share", "ellip_share",
    "discourse",
]


def extract_one(text: str) -> np.ndarray:
    tl = str(text).lower()
    toks = tokenize(tl)
    n = max(len(toks), 1)
    sents = split_sents(text)
    slens = [len(tokenize(s)) for s in sents]

    return np.array([
        np.log(n + 1),
        sum(1 for t in toks if t in PRONOUNS) / n,
        sum(1 for t in toks if t in STOP_RU) / n,
        np.mean(slens) if slens else 0,
        np.std(slens) if len(slens) > 1 else 0,
        len(set(toks)) / n,
        sum(c.isdigit() for c in text) / max(len(text), 1),
        sum(c.isascii() and c.isalpha() for c in text) / max(len(text), 1),
        text.count("?") / n,
        text.count("!") / n,
        text.count("...") / n,
        sum(tl.count(m) for m in DISCOURSE) / n,
    ])


def extract_all(texts) -> np.ndarray:
    return np.vstack([extract_one(t) for t in texts])


def style_markers(text: str) -> dict:
    toks = tokenize(str(text).lower())
    n = max(len(toks), 1)
    nc = max(len(text), 1)
    return {
        "pron_share": sum(1 for t in toks if t in PRONOUNS) / n,
        "excl_share": text.count("!") / n,
        "ques_share": text.count("?") / n,
        "elli_share": text.count("...") / n,
        "latin_share": sum(c.isascii() and c.isalpha() for c in text) / nc,
        "digit_share": sum(c.isdigit() for c in text) / nc,
        "stop_share": sum(1 for t in toks if t in STOP_RU) / n,
        "ttr": len(set(toks)) / n,
    }
