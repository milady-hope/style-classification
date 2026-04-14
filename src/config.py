import random
import numpy as np
import torch

SEED = 42
DATA_PATH = "data/data500.csv"

# ── preprocessing ──
TOKEN_RE_PAT = r"[A-Za-zА-Яа-яЁё0-9']+"
SENT_RE_PAT  = r"(?<=[.!?])\s+"

# ── baseline ──
PRONOUNS = {
    "я","меня","мне","мной", "ты","тебя","тебе","тобой",
    "вы","вас","вам","вами", "мы","нас","нам","нами",
    "он","его","ему","им", "она","её","ее","ей","ею",
    "они","их","им","ими", "себя","себе","собой",
}
DISCOURSE = [
    "рассмотрим","покажем","докажем","обозначим","пусть",
    "следовательно","итак","таким образом","однако",
    "например","в частности",
]

# ── SVM ──
SVM_C = 1.0
SVM_ANALYZER = "char"
SVM_NGRAM = (3, 6)
SVM_MIN_DF = 2
SVM_MAX_DF = 1.0
SVM_SUBLINEAR_TF = True

# ── Char-CNN-BiLSTM ──
EMB_DIM = 64
HIDDEN  = 256
DROPOUT = 0.3
MAX_CHAR_LEN = 1500
BATCH_SIZE = 32
EPOCHS_BI = 15
LR_BI = 3e-4
THRESH_BI = 0.40

# ── RuBERT classifier ──
BERT_MODEL = "DeepPavlov/rubert-base-cased"
MAX_LEN_BERT = 256
BS_BERT = 16
EP_BERT = 4
LR_BERT = 3e-5
WD_BERT = 0.01
WARMUP_RATIO = 0.1

# ── ruT5 generator ──
GEN_MODEL = "ai-forever/ruT5-base"
GEN_SEED = 24
PREFIX = "Приведи текст к научному стилю: "
MAX_SRC_LEN = 512
MAX_TGT_LEN = 512
GEN_LR = 3e-5
GEN_EPOCHS = 8
GEN_BATCH = 2
GEN_GRAD_ACC = 2
LABEL_SMOOTH = 0.1

# ── SBERT ──
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ── generation decoding ──
NUM_BEAMS = 4
LENGTH_PENALTY = 0.95
NO_REPEAT_NGRAM = 3
REP_PENALTY = 1.08
MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 20


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
