import os
import re
import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from src.config import (
    GEN_MODEL, PREFIX, MAX_SRC_LEN, MAX_TGT_LEN,
    GEN_LR, GEN_EPOCHS, GEN_BATCH, GEN_GRAD_ACC,
    LABEL_SMOOTH, GEN_SEED,
    NUM_BEAMS, LENGTH_PENALTY, NO_REPEAT_NGRAM,
    REP_PENALTY, MAX_NEW_TOKENS, MIN_NEW_TOKENS,
    get_device,
)


def load_generator(path=None):
    name = path or GEN_MODEL
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name).to(get_device())
    mdl.config.tie_word_embeddings = False
    return tok, mdl


def preprocess_fn(batch, tokenizer, prefix=PREFIX):
    inputs = [prefix + x for x in batch["source"]]
    model_inputs = tokenizer(inputs, max_length=MAX_SRC_LEN, truncation=True)
    labels = tokenizer(text_target=batch["target"], max_length=MAX_TGT_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_trainer(model, tokenizer, train_ds, val_ds, compute_metrics,
                  output_dir="style_transfer_rut5", best_metric="style_content_lang_score",
                  greater_is_better=True, seed=GEN_SEED):
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=GEN_LR,
        weight_decay=0.01,
        warmup_steps=20,
        max_grad_norm=1.0,
        label_smoothing_factor=LABEL_SMOOTH,
        per_device_train_batch_size=GEN_BATCH,
        per_device_eval_batch_size=GEN_BATCH,
        gradient_accumulation_steps=GEN_GRAD_ACC,
        num_train_epochs=GEN_EPOCHS,
        predict_with_generate=False,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=greater_is_better,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=seed,
    )
    return Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )


@torch.no_grad()
def generate_texts(texts, tokenizer, model, prefix=PREFIX, device=None):
    device = device or get_device()
    model.eval()
    model.generation_config.max_length = None
    model.generation_config.min_length = None
    results = []
    for t in texts:
        enc = tokenizer(
            prefix + t, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN,
        ).to(device)
        ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            repetition_penalty=REP_PENALTY,
            early_stopping=True,
        )
        results.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return results
