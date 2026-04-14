from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.preprocessing import clean
from src.config import MAX_CHAR_LEN

PAD_ID = 0
UNK_ID = 1


def build_char_vocab(texts, min_freq=2):
    c = Counter()
    for t in texts:
        c.update(list(clean(t).lower()))
    v = {"<PAD>": 0, "<UNK>": 1}
    for ch, freq in c.most_common():
        if freq >= min_freq:
            v[ch] = len(v)
    return v


def text_to_ids(text, vocab, max_len=MAX_CHAR_LEN):
    t = clean(text).lower()
    ids = [vocab.get(c, UNK_ID) for c in t[:max_len]]
    length = len(ids)
    ids += [PAD_ID] * (max_len - length)
    return ids, length


class CharDataset(Dataset):
    def __init__(self, df, vocab, max_len=MAX_CHAR_LEN):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.ml = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids, ln = text_to_ids(self.texts[i], self.vocab, self.ml)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(ln, dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


class _Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)

    def forward(self, x, lengths=None):
        sc = self.w(x).squeeze(-1)
        if lengths is not None:
            mask = (
                torch.arange(x.size(1), device=x.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )
            sc = sc.masked_fill(mask, -1e9)
        wt = torch.softmax(sc, dim=1).unsqueeze(-1)
        return (x * wt).sum(dim=1)


class CharCNNBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=256, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(emb, 128, k, padding=k // 2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(4),
            )
            for k in [3, 5, 7]
        ])
        self.lstm = nn.LSTM(384, hid, batch_first=True, bidirectional=True)
        self.attn = _Attention(hid * 2)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hid * 2, 2)

    def forward(self, ids, lengths=None):
        x = self.emb(ids).permute(0, 2, 1)
        x = torch.cat([c(x) for c in self.convs], dim=1).permute(0, 2, 1)
        lens_adj = (lengths // 4).clamp(min=1) if lengths is not None else None
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.attn(out, lens_adj)))
