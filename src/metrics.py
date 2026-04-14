import re
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, f1_score,
)

from src.config import SBERT_MODEL, get_device

_word_re = re.compile(r"\w+", flags=re.UNICODE)


def _count_words(text: str) -> int:
    return max(1, len(_word_re.findall(text)))


# ── Classification ──

def clf_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=3)


def clf_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {"acc": acc, "prec": p, "rec": r, "f1": f1}


def conf_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


# ── SBERT ──

_sbert_model = None


def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(SBERT_MODEL, device="cpu")
    return _sbert_model


def sbert_encode(texts, batch_size=32):
    return _get_sbert().encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )


# ── Style classifier proxy ──

@torch.no_grad()
def p_scientific_batch(texts, clf_tokenizer, clf_model, batch_size=32, max_len=256):
    device = get_device()
    clf_model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        x = clf_tokenizer(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=max_len,
        ).to(device)
        logits = clf_model(**x).logits
        p1 = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        probs.append(p1)
    return np.concatenate(probs)


# ── Language quality ──

_lt_tool = None


def _get_lt():
    global _lt_tool
    if _lt_tool is None:
        import language_tool_python
        _lt_tool = language_tool_python.LanguageTool("ru-RU")
    return _lt_tool


def count_language_errors(texts):
    tool = _get_lt()
    errs, errs100 = [], []
    for text in texts:
        try:
            n = len(tool.check(text))
        except Exception:
            n = 0
        w = _count_words(text)
        errs.append(n)
        errs100.append(100.0 * n / w)
    return np.array(errs, dtype=float), np.array(errs100, dtype=float)


# ── Generation evaluation ──

def evaluate_generation(src_texts, tgt_texts, pred_texts,
                        clf_tokenizer, clf_model, label="MODEL"):
    import sacrebleu

    bleu_metric = sacrebleu.metrics.BLEU(max_ngram_order=4, smooth_method="exp")
    bleu_res = bleu_metric.corpus_score(pred_texts, [tgt_texts])

    p_src = p_scientific_batch(src_texts, clf_tokenizer, clf_model)
    p_pred = p_scientific_batch(pred_texts, clf_tokenizer, clf_model)

    style_acc = float((p_pred >= 0.5).mean())
    delta_p = float((p_pred - p_src).mean())

    emb_src = sbert_encode(src_texts)
    emb_tgt = sbert_encode(tgt_texts)
    emb_pred = sbert_encode(pred_texts)

    sim_src = (emb_src * emb_pred).sum(axis=1)
    sim_tgt = (emb_tgt * emb_pred).sum(axis=1)

    err_cnt, err_100 = count_language_errors(pred_texts)

    return {
        "label": label,
        "BLEU": float(bleu_res.score),
        "Style accuracy": style_acc,
        "Mean p(scientific) source": float(p_src.mean()),
        "Mean p(scientific) pred": float(p_pred.mean()),
        "Mean Δp(scientific)": delta_p,
        "SBERT src-pred mean": float(sim_src.mean()),
        "SBERT tgt-pred mean": float(sim_tgt.mean()),
        "Lang errors/100 words mean": float(err_100.mean()),
    }


# ── Composite metric for Seq2SeqTrainer ──

def make_gen_compute_metrics(gen_tokenizer, clf_tokenizer, clf_model,
                             val_src_emb, val_tgt_emb):
    import evaluate as hf_eval
    bleu_m = hf_eval.load("sacrebleu")

    vocab_size = len(gen_tokenizer)
    pad_id = gen_tokenizer.pad_token_id or gen_tokenizer.eos_token_id or 0

    def _sanitize(x):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3:
            x = x.argmax(axis=-1)
        if np.issubdtype(x.dtype, np.floating):
            x = np.rint(x).astype(np.int64)
        else:
            x = x.astype(np.int64, copy=False)
        return np.where((x >= 0) & (x < vocab_size), x, pad_id)

    def compute(eval_pred):
        preds, labels = eval_pred
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, pad_id).astype(np.int64)

        try:
            dec_preds = gen_tokenizer.batch_decode(
                _sanitize(preds).tolist(), skip_special_tokens=True)
            dec_labels = gen_tokenizer.batch_decode(
                labels.tolist(), skip_special_tokens=True)
        except Exception:
            return {"style_content_lang_score": -1.0}

        pred_emb = sbert_encode(dec_preds)
        sim_tgt = (val_tgt_emb * pred_emb).sum(axis=1)
        sim_tgt_mean = float(sim_tgt.mean())

        p_pred = p_scientific_batch(dec_preds, clf_tokenizer, clf_model)
        style_acc = float((p_pred >= 0.5).mean())

        bleu = bleu_m.compute(
            predictions=dec_preds,
            references=[[x] for x in dec_labels],
        )["score"] / 100.0

        score = 0.5 * sim_tgt_mean + 0.3 * style_acc + 0.2 * bleu

        result = {
            "cos_tgt_pred_mean": sim_tgt_mean,
            "style_acc_scientific": style_acc,
            "p_scientific_mean": float(p_pred.mean()),
            "bleu": float(bleu),
            "style_content_lang_score": float(score),
        }

        if val_src_emb is not None:
            sim_src = (val_src_emb * pred_emb).sum(axis=1)
            result["cos_src_pred_mean"] = float(sim_src.mean())

        return result

    return compute
