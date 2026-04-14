"""
Преобразование стиля: ruT5 + RuBERT classifier.
  - Обучение классификатора стиля (RuBERT)
  - Обучение генератора (ruT5-base)
  - Сравнение 4 конфигураций: с/без префикса x style_transfer/eval_loss
  - Устойчивость по 3 random seed
  - Бейзлайн (копирование входа)
  - Анализ ошибок
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
import evaluate as hf_eval

from src.config import (
    GEN_SEED, SEED, get_device,
    BERT_MODEL, GEN_MODEL, PREFIX,
    MAX_SRC_LEN, MAX_TGT_LEN,
    GEN_LR, GEN_EPOCHS, GEN_BATCH, GEN_GRAD_ACC, LABEL_SMOOTH,
)
from src.preprocessing import load_data, build_pairs_df
from src.models.generator import load_generator, preprocess_fn, generate_texts
from src.metrics import (
    sbert_encode, p_scientific_batch,
    evaluate_generation, make_gen_compute_metrics,
)

device = get_device()


# ═══════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════

def train_or_load_classifier(df, train_ids, val_ids):
    clf_dir = "style_clf_rubert/best"
    if os.path.isdir(clf_dir) and os.path.isfile(os.path.join(clf_dir, "config.json")):
        tok = AutoTokenizer.from_pretrained(clf_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(clf_dir).to(device).eval()
        return tok, mdl

    clf_df = df[["text", "label", "pair_id"]].dropna().rename(columns={"label": "labels"}).reset_index(drop=True)
    train_clf = HFDataset.from_pandas(clf_df[clf_df.pair_id.isin(train_ids)][["text", "labels"]].reset_index(drop=True), preserve_index=False)
    val_clf = HFDataset.from_pandas(clf_df[clf_df.pair_id.isin(val_ids)][["text", "labels"]].reset_index(drop=True), preserve_index=False)

    tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2).to(device)

    def _prep(batch): return tok(batch["text"], truncation=True, max_length=256)
    train_tok = train_clf.map(_prep, batched=True, remove_columns=["text"])
    val_tok = val_clf.map(_prep, batched=True, remove_columns=["text"])

    acc_m, f1_m = hf_eval.load("accuracy"), hf_eval.load("f1")
    def _metrics(ep):
        logits, labels = ep
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": acc_m.compute(predictions=preds, references=labels)["accuracy"],
                "f1": f1_m.compute(predictions=preds, references=labels, average="macro")["f1"]}

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    args = TrainingArguments(
        output_dir="style_clf_rubert", eval_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=4, weight_decay=0.01, logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        fp16=torch.cuda.is_available() and not use_bf16, bf16=use_bf16,
        report_to="none", save_total_limit=2, seed=GEN_SEED)
    trainer = Trainer(model=mdl, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=_metrics)
    trainer.train()
    os.makedirs(clf_dir, exist_ok=True)
    trainer.save_model(clf_dir); tok.save_pretrained(clf_dir)
    tok = AutoTokenizer.from_pretrained(clf_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(clf_dir).to(device).eval()
    return tok, mdl


# ═══════════════════════════════════════════════════════════════════════════
# Train one generator config
# ═══════════════════════════════════════════════════════════════════════════

def train_one_config(train_ds, val_ds, val_df, clf_tok, clf_model,
                     prefix, best_metric, greater, out_dir, seed=GEN_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(device)
    gen_mdl.config.tie_word_embeddings = False

    def _preprocess(batch):
        inputs = [prefix + x for x in batch["source"]]
        mi = gen_tok(inputs, max_length=MAX_SRC_LEN, truncation=True)
        labels = gen_tok(text_target=batch["target"], max_length=MAX_TGT_LEN, truncation=True)
        mi["labels"] = labels["input_ids"]
        return mi

    train_tok_ds = train_ds.map(_preprocess, batched=True, remove_columns=["source", "target"])
    val_tok_ds = val_ds.map(_preprocess, batched=True, remove_columns=["source", "target"])

    val_src_emb = sbert_encode(val_df["source"].tolist())
    val_tgt_emb = sbert_encode(val_df["target"].tolist())
    compute_metrics = make_gen_compute_metrics(gen_tok, clf_tok, clf_model, val_src_emb, val_tgt_emb)

    collator = DataCollatorForSeq2Seq(gen_tok, model=gen_mdl, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, eval_strategy="epoch", save_strategy="epoch",
        learning_rate=GEN_LR, weight_decay=0.01, warmup_steps=20, max_grad_norm=1.0,
        label_smoothing_factor=LABEL_SMOOTH, per_device_train_batch_size=GEN_BATCH,
        per_device_eval_batch_size=GEN_BATCH, gradient_accumulation_steps=GEN_GRAD_ACC,
        num_train_epochs=GEN_EPOCHS, predict_with_generate=False, logging_steps=10,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model=best_metric, greater_is_better=greater,
        bf16=use_bf16, fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none", seed=seed)
    trainer = Seq2SeqTrainer(model=gen_mdl, args=args, train_dataset=train_tok_ds,
                              eval_dataset=val_tok_ds, data_collator=collator, compute_metrics=compute_metrics)
    trainer.train()
    gen_dir = f"{out_dir}/best"
    trainer.save_model(gen_dir); gen_tok.save_pretrained(gen_dir)
    del trainer; torch.cuda.empty_cache()
    return gen_dir, gen_tok


# ═══════════════════════════════════════════════════════════════════════════
# 4 configs comparison
# ═══════════════════════════════════════════════════════════════════════════

def compare_4_configs(train_ds, val_ds, val_df, test_df, clf_tok, clf_model):
    CONFIGS = [
        {"name": "Prefix + style_transfer",    "prefix": PREFIX, "metric": "style_content_lang_score", "greater": True},
        {"name": "Prefix + eval_loss",          "prefix": PREFIX, "metric": "eval_loss",                "greater": False},
        {"name": "No prefix + style_transfer",  "prefix": "",     "metric": "style_content_lang_score", "greater": True},
        {"name": "No prefix + eval_loss",       "prefix": "",     "metric": "eval_loss",                "greater": False},
    ]
    src_texts, tgt_texts = test_df["source"].tolist(), test_df["target"].tolist()
    results = []

    for cfg in CONFIGS:
        print(f"\n{'='*60}\n{cfg['name']}\n{'='*60}")
        safe = cfg["name"].replace(" ", "_").replace("+", "")
        gen_dir, gen_tok = train_one_config(
            train_ds, val_ds, val_df, clf_tok, clf_model,
            cfg["prefix"], cfg["metric"], cfg["greater"], f"config_{safe}")

        gen_mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(device).eval()
        preds = generate_texts(src_texts, gen_tok, gen_mdl, prefix=cfg["prefix"])
        res = evaluate_generation(src_texts, tgt_texts, preds, clf_tok, clf_model, label=cfg["name"])
        res["config"] = cfg["name"]
        results.append(res)
        del gen_mdl; torch.cuda.empty_cache()

    print("\n=== 4 Configs Summary ===")
    df = pd.DataFrame(results)
    print(f"{'Config':<40} {'CosSim':>8} {'Accuracy':>10} {'BLEU':>8}")
    for _, r in df.iterrows():
        print(f"{r['config']:<40} {r['SBERT tgt-pred mean']:>8.4f} {r['Style accuracy']:>10.4f} {r['BLEU']:>8.2f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed stability
# ═══════════════════════════════════════════════════════════════════════════

def multi_seed_runs(pairs, clf_tok, clf_model):
    SEEDS = [24, 42, 7]
    results = []

    for seed_i in SEEDS:
        print(f"\n{'='*60}\nSEED = {seed_i}\n{'='*60}")
        random.seed(seed_i); np.random.seed(seed_i)
        torch.manual_seed(seed_i); torch.cuda.manual_seed_all(seed_i)

        pair_ids = pairs["pair_id"].unique()
        train_ids, tmp_ids = train_test_split(pair_ids, test_size=0.30, random_state=seed_i)
        val_ids, test_ids = train_test_split(tmp_ids, test_size=0.66667, random_state=seed_i)

        train_df = pairs[pairs.pair_id.isin(train_ids)].reset_index(drop=True)
        val_df = pairs[pairs.pair_id.isin(val_ids)].reset_index(drop=True)
        test_df = pairs[pairs.pair_id.isin(test_ids)].reset_index(drop=True)

        train_ds = HFDataset.from_pandas(train_df[["source", "target"]], preserve_index=False)
        val_ds = HFDataset.from_pandas(val_df[["source", "target"]], preserve_index=False)

        gen_dir, gen_tok = train_one_config(
            train_ds, val_ds, val_df, clf_tok, clf_model,
            PREFIX, "style_content_lang_score", True, f"seed_{seed_i}", seed=seed_i)

        gen_mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(device).eval()
        src, tgt = test_df["source"].tolist(), test_df["target"].tolist()
        preds = generate_texts(src, gen_tok, gen_mdl)
        res = evaluate_generation(src, tgt, preds, clf_tok, clf_model, label=f"SEED={seed_i}")
        res["seed"] = seed_i
        results.append(res)
        del gen_mdl; torch.cuda.empty_cache()

    print("\n=== Multi-seed Summary ===")
    df = pd.DataFrame(results)
    for col in ["Style accuracy", "SBERT tgt-pred mean", "BLEU"]:
        vals = df[col]
        print(f"  {col}: {vals.mean():.4f} +- {vals.std():.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    random.seed(GEN_SEED); np.random.seed(GEN_SEED)
    torch.manual_seed(GEN_SEED); torch.cuda.manual_seed_all(GEN_SEED)

    df = load_data()
    pairs = build_pairs_df(df).sample(frac=1.0, random_state=GEN_SEED).reset_index(drop=True)

    pair_ids = pairs["pair_id"].unique()
    train_ids, tmp_ids = train_test_split(pair_ids, test_size=0.30, random_state=GEN_SEED)
    val_ids, test_ids = train_test_split(tmp_ids, test_size=0.66667, random_state=GEN_SEED)

    train_df = pairs[pairs.pair_id.isin(train_ids)].reset_index(drop=True)
    val_df = pairs[pairs.pair_id.isin(val_ids)].reset_index(drop=True)
    test_df = pairs[pairs.pair_id.isin(test_ids)].reset_index(drop=True)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_ds = HFDataset.from_pandas(train_df[["source", "target"]], preserve_index=False)
    val_ds = HFDataset.from_pandas(val_df[["source", "target"]], preserve_index=False)

    # Classifier
    clf_tok, clf_model = train_or_load_classifier(df, train_ids, val_ids)

    # Main generator
    print("\n" + "=" * 70 + "\nMAIN GENERATOR (prefix + style_transfer)\n" + "=" * 70)
    gen_dir, gen_tok = train_one_config(
        train_ds, val_ds, val_df, clf_tok, clf_model,
        PREFIX, "style_content_lang_score", True, "style_transfer_rut5")

    gen_mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(device).eval()
    src_texts, tgt_texts = test_df["source"].tolist(), test_df["target"].tolist()
    pred_texts = generate_texts(src_texts, gen_tok, gen_mdl)

    res_main = evaluate_generation(src_texts, tgt_texts, pred_texts, clf_tok, clf_model, label="ruT5 main")

    # Baseline: copy
    res_bl = evaluate_generation(src_texts, tgt_texts, src_texts, clf_tok, clf_model, label="BASELINE (copy)")

    print("\n=== Model vs Baseline ===")
    print(f"{'Metric':<30} {'Baseline':>12} {'Model':>12} {'Delta':>12}")
    for k in ["Style accuracy", "Mean p(scientific) pred", "SBERT tgt-pred mean", "BLEU"]:
        print(f"{k:<30} {res_bl[k]:>12.4f} {res_main[k]:>12.4f} {res_main[k]-res_bl[k]:>+12.4f}")

    del gen_mdl; torch.cuda.empty_cache()

    # 4 configs
    print("\n" + "=" * 70 + "\n4 CONFIGURATIONS COMPARISON\n" + "=" * 70)
    config_results = compare_4_configs(train_ds, val_ds, val_df, test_df, clf_tok, clf_model)

    # Multi-seed
    print("\n" + "=" * 70 + "\nMULTI-SEED STABILITY\n" + "=" * 70)
    seed_results = multi_seed_runs(pairs, clf_tok, clf_model)

    # Final summary
    print("\n" + "=" * 70 + "\nFINAL SUMMARY\n" + "=" * 70)
    print(f"Corpus: 500 pairs")
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Generator: ruT5-base")
    print(f"Classifier: RuBERT")
    for k, v in res_main.items():
        if k != "label":
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
