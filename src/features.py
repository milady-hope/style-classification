import re
import numpy as np


PRONOUNS = {
    "я", "меня", "мне", "мной",
    "ты", "тебя", "тебе", "тобой",
    "вы", "вас", "вам", "вами",
    "он", "его", "ему", "им",
    "она", "её", "ее", "ей", "ею",
    "оно",
    "мы", "нас", "нам", "нами",
    "они", "их", "ими",
    "себя", "себе", "собой",
}

DISCOURSE_MARKERS = [
    "рассмотрим", "покажем", "докажем", "обозначим", "пусть",
    "следовательно", "итак", "таким образом", "однако", "впрочем",
    "например", "в частности", "с другой стороны", "заметим", "поэтому",
    "давайте", "представим",
]

_token_re = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
_sent_split_re = re.compile(r"[.!?…]+")
_latin_re = re.compile(r"[A-Za-z]")
_cyril_re = re.compile(r"[А-Яа-яЁё]")
_digit_re = re.compile(r"\d")


def _tokenize(text: str):
    return _token_re.findall(text.lower())


def _split_sentences(text: str):
    sents = [s.strip() for s in _sent_split_re.split(text) if s.strip()]
    return sents if sents else [text.strip()]


def _count_marker(text_lc: str, marker: str) -> int:
    if " " in marker:
        return text_lc.count(marker)
    return len(re.findall(rf"\b{re.escape(marker)}\b", text_lc))


def extract_one(text: str) -> np.ndarray:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    tl = text.lower()
    tokens = _tokenize(tl)
    n_tokens = max(len(tokens), 1)
    n_chars = max(len(text), 1)

    text_len_norm = np.log(n_tokens + 1)
    pron_share = sum(1 for t in tokens if t in PRONOUNS) / n_tokens

    sents = _split_sentences(text)
    sent_lens = [len(_tokenize(s)) for s in sents]
    avg_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0
    std_sent_len = float(np.std(sent_lens)) if sent_lens else 0.0

    excl_share = text.count("!") / n_chars
    ques_share = text.count("?") / n_chars
    elli_share = (text.count("…") + text.count("...")) / n_chars

    latin_cnt = len(_latin_re.findall(text))
    cyril_cnt = len(_cyril_re.findall(text))
    letters_total = max(latin_cnt + cyril_cnt, 1)
    latin_share = latin_cnt / letters_total
    digit_share = len(_digit_re.findall(text)) / n_chars
    RUS_STOPWORDS = "stopwords"
    stop_share = sum(1 for t in tokens if t in RUS_STOPWORDS) / n_tokens

    ttr = len(set(tokens)) / n_tokens

    marker_counts = [_count_marker(tl, m) for m in DISCOURSE_MARKERS]
    marker_sum = sum(marker_counts)
    marker_rate = marker_sum / n_tokens
    marker_any = 1.0 if marker_sum > 0 else 0.0

    return np.array([
        pron_share, excl_share, ques_share, elli_share,
        latin_share, digit_share, marker_rate, marker_any,
        avg_sent_len, std_sent_len, text_len_norm, stop_share, ttr,
    ], dtype=np.float32)


def extract_all(texts) -> np.ndarray:
    if hasattr(texts, "values"):
        texts = texts.values
    return np.vstack([extract_one(t) for t in texts])


FEAT_NAMES = [
    "pron_share", "excl_share", "ques_share", "elli_share",
    "latin_share", "digit_share", "marker_rate", "marker_any",
    "avg_sent_len", "std_sent_len", "text_len_norm", "stop_share", "ttr",
]

_PRON_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted(PRONOUNS, key=len, reverse=True))) + r")\b",
    re.IGNORECASE,
)


def ablate_no_pronouns(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = _PRON_RE.sub(" ", text)
    return re.sub(r"\s+", " ", t).strip()


def ablate_no_punct(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.replace("...", "…")
    t = re.sub(r"[!?…]+", " ", t)
    t = t.replace("«", "").replace("»", "")
    return re.sub(r"\s+", " ", t).strip()


def ablate_clip_len(text: str, max_tokens: int = 200) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    toks = _token_re.findall(text)
    if len(toks) <= max_tokens:
        return text
    return " ".join(toks[:max_tokens])
