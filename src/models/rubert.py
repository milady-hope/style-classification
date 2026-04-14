import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.config import (
    BERT_MODEL, MAX_LEN_BERT, BS_BERT, EP_BERT,
    LR_BERT, WD_BERT, WARMUP_RATIO, get_device,
)


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN_BERT):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.ml = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.ml,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_MODEL)


def load_model(device=None):
    device = device or get_device()
    return AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=2
    ).to(device)


def evaluate(model, dl, device=None):
    device = device or get_device()
    model.eval()
    preds, labels = [], []
    tot_loss = 0
    with torch.no_grad():
        for b in dl:
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device),
            )
            tot_loss += out.loss.item()
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            labels.extend(b["labels"].tolist())
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"acc": acc, "prec": p, "rec": r, "f1": f1, "loss": tot_loss / len(dl)}, preds


def train_one_epoch(model, dl, optimizer, scheduler, device=None):
    device = device or get_device()
    model.train()
    for b in dl:
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=b["input_ids"].to(device),
            attention_mask=b["attention_mask"].to(device),
            labels=b["labels"].to(device),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
