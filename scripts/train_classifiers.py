import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.config import (
    SEED, set_seed, get_device,
    EMB_DIM, HIDDEN, DROPOUT, MAX_CHAR_LEN,
    BATCH_SIZE, EPOCHS_BI, LR_BI, THRESH_BI,
    BS_BERT, EP_BERT, LR_BERT, WD_BERT, WARMUP_RATIO,
    BERT_MODEL,
)
from src.preprocessing import load_data, train_test_by_pairs, group_kfold
from src.features import extract_all, FEAT_NAMES
from src.models.baseline import build_baseline
from src.models.bilstm import CharCNNBiLSTM, CharDataset, build_char_vocab
from src.models.rubert import (
    BertDataset, load_tokenizer, load_model as load_bert,
    evaluate as eval_bert, train_one_epoch,
)
from src.metrics import clf_metrics

from transformers import get_linear_schedule_with_warmup
import shap

set_seed()
device = get_device()


def baseline_grid_search(train_df):
    X = extract_all(train_df["text_clean"])
    y = train_df["label"].values
    groups = train_df["pair_id"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED)),
    ])
    param_grid = {
        "lr__C": [0.01, 0.1, 1.0, 10.0],
        "lr__class_weight": [None, "balanced"],
    }
    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, param_grid, cv=gkf, scoring="f1_macro", refit=True, n_jobs=-1)
    gs.fit(X, y, groups=groups)

    print("Baseline GridSearch")
    res = pd.DataFrame(gs.cv_results_)
    for _, row in res.iterrows():
        print(f"  C={row['param_lr__C']}, cw={row['param_lr__class_weight']}, "
              f"F1={row['mean_test_score']:.4f}+-{row['std_test_score']:.4f}")
    print(f"Best: {gs.best_params_}, F1={gs.best_score_:.4f}")
    return gs.best_estimator_


def baseline_ablation(train_df, test_df):
    # Для каждого режима абляции: обучаем на ablated train, оцениваем на ablated test.
    # Тест фиксирован — сравнение корректное.
    def _run(mode):
        def _apply(texts):
            if mode == "no_pron":
                X = extract_all(texts)
                X[:, 1] = 0   # обнуляем признак доли местоимений
                return X
            elif mode == "no_punct":
                X = extract_all(texts)
                X[:, 8:11] = 0  # обнуляем признаки пунктуации
                return X
            elif mode == "trunc200":
                return extract_all(texts.apply(lambda t: " ".join(t.split()[:200])))
            else:
                return extract_all(texts)

        X_tr = _apply(train_df["text_clean"])
        X_te = _apply(test_df["text_clean"])
        y_tr = train_df["label"].values
        y_te = test_df["label"].values

        pipe = Pipeline([("s", StandardScaler()),
                         ("lr", LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED))])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        return {"acc": accuracy_score(y_te, p), "f1": f1_score(y_te, p, average="macro")}

    print("\nAblation (train → test)")
    for label, mode in [("Full", "full"), ("No pronouns", "no_pron"),
                         ("No punctuation", "no_punct"), ("Trunc 200", "trunc200")]:
        r = _run(mode)
        print(f"  {label:<25} Acc={r['acc']:.3f}  F1={r['f1']:.3f}")


def baseline_coefficients(pipe, X, y):
    pipe.fit(X, y)
    coefs = pipe.named_steps["lr"].coef_[0]
    ct = pd.DataFrame({"feat": FEAT_NAMES, "coef": coefs})
    ct = ct.reindex(ct["coef"].abs().sort_values(ascending=False).index)
    print("\nCoefficients")
    for _, row in ct.iterrows():
        print(f"  {row['feat']:<20} {row['coef']:+.6f}")


def run_baseline(data, train_df, test_df):
    best_pipe = baseline_grid_search(train_df)   # ← только train
    baseline_ablation(train_df, test_df)          # ← train → test

    X_tr = extract_all(train_df["text_clean"])
    y_tr = train_df["label"].values
    groups_tr = train_df["pair_id"].values
    baseline_coefficients(best_pipe, X_tr, y_tr)

    gkf = GroupKFold(n_splits=5)
    folds = []
    for fold, (tr, te) in enumerate(gkf.split(X_tr, y_tr, groups_tr), 1):  # ← только train
        best_pipe.fit(X_tr[tr], y_tr[tr])
        pred = best_pipe.predict(X_tr[te])
        acc = accuracy_score(y_tr[te], pred)
        p, r, f1, _ = precision_recall_fscore_support(y_tr[te], pred, average="macro")
        folds.append({"fold": fold, "acc": acc, "prec": p, "rec": r, "f1": f1})
    cv = pd.DataFrame(folds)
    print("\nBaseline CV (только train)")
    print(cv[["fold", "acc", "f1"]].to_string(index=False))
    print(f"Mean: Acc={cv['acc'].mean():.3f}+-{cv['acc'].std():.3f}, F1={cv['f1'].mean():.3f}+-{cv['f1'].std():.3f}")

    X_te = extract_all(test_df["text_clean"])
    best_pipe.fit(X_tr, y_tr)
    y_pred = best_pipe.predict(X_te)
    print("\n" + classification_report(test_df["label"].values, y_pred, digits=3))
    return y_pred, cv


def svm_grid_search(train_df):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", sublinear_tf=True, norm="l2")),
        ("clf", LinearSVC(max_iter=1000)),
    ])
    param_grid = {
        "tfidf__ngram_range": [(3, 5), (3, 6)],
        "tfidf__min_df": [2, 3, 5],
        "clf__C": [0.1, 0.5, 1.0, 5.0],
    }
    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, param_grid, cv=gkf, scoring="f1_macro", refit=True, n_jobs=-1)
    gs.fit(train_df["text_clean"], train_df["label"], groups=train_df["pair_id"])  # ← только train

    print("SVM GridSearch")
    res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    for _, row in res.head(10).iterrows():
        print(f"  ngram={row['param_tfidf__ngram_range']}, min_df={row['param_tfidf__min_df']}, "
              f"C={row['param_clf__C']}, F1={row['mean_test_score']:.4f}+-{row['std_test_score']:.4f}")
    print(f"Best: {gs.best_params_}, F1={gs.best_score_:.4f}")
    return gs.best_estimator_


def svm_shap_analysis(pipe, train_df, test_df):
    vct = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    X_tr = vct.transform(train_df["text_clean"])
    X_te = vct.transform(test_df["text_clean"])
    feat_names = np.array(vct.get_feature_names_out())

    expl = shap.LinearExplainer(clf, X_tr)
    shap_vals = expl.shap_values(X_te)
    mean_shap = shap_vals.mean(axis=0)

    top_sci = np.argsort(mean_shap)[-10:][::-1]
    top_pop = np.argsort(mean_shap)[:10]
    print("\nSHAP SVM: scientific")
    for i in top_sci:
        print(f"  {feat_names[i]:<20} SHAP={mean_shap[i]:+.4f}")
    print("SHAP SVM: popular")
    for i in top_pop:
        print(f"  {feat_names[i]:<20} SHAP={mean_shap[i]:+.4f}")


def run_svm(data, train_df, test_df):
    best_pipe = svm_grid_search(train_df)   # ← только train

    gkf = GroupKFold(n_splits=5)
    folds = []
    for fold, (tr, te) in enumerate(gkf.split(train_df["text_clean"], train_df["label"], groups=train_df["pair_id"]), 1):  # ← только train
        best_pipe.fit(train_df.iloc[tr]["text_clean"], train_df.iloc[tr]["label"])
        preds = best_pipe.predict(train_df.iloc[te]["text_clean"])
        acc = accuracy_score(train_df.iloc[te]["label"], preds)
        p, r, f1, _ = precision_recall_fscore_support(train_df.iloc[te]["label"], preds, average="macro")
        folds.append({"fold": fold, "acc": acc, "prec": p, "rec": r, "f1": f1})
    cv = pd.DataFrame(folds)
    print("\nSVM CV (только train)")
    print(cv[["fold", "acc", "f1"]].to_string(index=False))
    print(f"Mean: F1={cv['f1'].mean():.4f}+-{cv['f1'].std():.4f}")

    best_pipe.fit(train_df["text_clean"], train_df["label"])
    y_pred = best_pipe.predict(test_df["text_clean"])
    print("\n" + classification_report(test_df["label"].values, y_pred, digits=3))

    svm_shap_analysis(best_pipe, train_df, test_df)
    return y_pred, cv


def _bilstm_one_fold(fold_tr, fold_val, emb, hid, drop, lr,
                     epochs=EPOCHS_BI, max_clen=MAX_CHAR_LEN):
    vocab = build_char_vocab(fold_tr["text"].tolist())
    tr_dl = DataLoader(CharDataset(fold_tr, vocab, max_clen), batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(CharDataset(fold_val, vocab, max_clen), batch_size=BATCH_SIZE)

    model = CharCNNBiLSTM(len(vocab), emb, hid, drop).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_f1, pat = 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tr_dl:
            opt.zero_grad()
            crit(model(b["input_ids"].to(device), b["lengths"].to(device)), b["labels"].to(device)).backward()
            opt.step()
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for b in va_dl:
                lg = model(b["input_ids"].to(device), b["lengths"].to(device))
                preds.extend((torch.softmax(lg, -1)[:, 1] >= THRESH_BI).long().cpu().tolist())
                labels.extend(b["labels"].tolist())
        f1_v = f1_score(labels, preds, average="macro")
        sched.step(f1_v)
        if f1_v > best_f1:
            best_f1, pat = f1_v, 0
        else:
            pat += 1
            if pat >= 4: break

    acc = accuracy_score(labels, preds)
    p, r, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"acc": acc, "prec": p, "rec": r, "f1": f1_m}


def bilstm_hyperparam_search(train_df):
    configs = [
        {"emb": 32,  "hid": 128, "drop": 0.3, "lr": 3e-4, "epochs": 15},
        {"emb": 64,  "hid": 256, "drop": 0.3, "lr": 3e-4, "epochs": 15},
        {"emb": 64,  "hid": 256, "drop": 0.3, "lr": 3e-4, "epochs": 20},
        {"emb": 64,  "hid": 256, "drop": 0.3, "lr": 3e-4, "epochs": 25},
        {"emb": 64,  "hid": 256, "drop": 0.2, "lr": 1e-4, "epochs": 20},
        {"emb": 64,  "hid": 256, "drop": 0.5, "lr": 5e-4, "epochs": 15},
        {"emb": 128, "hid": 512, "drop": 0.3, "lr": 3e-4, "epochs": 20},
    ]
    print("BiLSTM Hyperparam Search")
    best_f1, best_cfg = 0, configs[1]
    gkf = GroupKFold(n_splits=5)
    for cfg in configs:
        fold_f1s = []
        for fold, (tr_i, val_i) in enumerate(gkf.split(np.zeros(len(train_df)), train_df["label"], groups=train_df["pair_id"]), 1):
            res = _bilstm_one_fold(train_df.iloc[tr_i].reset_index(drop=True),
                                    train_df.iloc[val_i].reset_index(drop=True),
                                    cfg["emb"], cfg["hid"], cfg["drop"], cfg["lr"],
                                    epochs=cfg["epochs"])
            fold_f1s.append(res["f1"])
        mf = np.mean(fold_f1s)
        print(f"  emb={cfg['emb']}, hid={cfg['hid']}, drop={cfg['drop']}, "
              f"lr={cfg['lr']}, ep={cfg['epochs']}: F1={mf:.4f}+-{np.std(fold_f1s):.4f}")
        if mf > best_f1: best_f1, best_cfg = mf, cfg
    print(f"Best: {best_cfg}, F1={best_f1:.4f}")
    return best_cfg


def run_bilstm(data, train_df, test_df):
    best = bilstm_hyperparam_search(train_df)

    gkf = GroupKFold(n_splits=5)
    folds = []
    for fold, (tr_i, val_i) in enumerate(gkf.split(np.zeros(len(train_df)), train_df["label"], groups=train_df["pair_id"]), 1):
        res = _bilstm_one_fold(train_df.iloc[tr_i].reset_index(drop=True),
                                train_df.iloc[val_i].reset_index(drop=True),
                                best["emb"], best["hid"], best["drop"], best["lr"],
                                epochs=best["epochs"])
        folds.append({"fold": fold, **res})
        print(f"  Fold {fold}: acc={res['acc']:.3f}, f1={res['f1']:.3f}")
    cv = pd.DataFrame(folds)
    print(f"BiLSTM: Acc={cv['acc'].mean():.3f}+-{cv['acc'].std():.3f}, F1={cv['f1'].mean():.3f}+-{cv['f1'].std():.3f}")

    vocab_fin = build_char_vocab(train_df["text"].tolist())
    tr_dl = DataLoader(CharDataset(train_df, vocab_fin), batch_size=BATCH_SIZE, shuffle=True)
    te_dl = DataLoader(CharDataset(test_df, vocab_fin), batch_size=BATCH_SIZE)
    m = CharCNNBiLSTM(len(vocab_fin), best["emb"], best["hid"], best["drop"]).to(device)
    opt = AdamW(m.parameters(), lr=best["lr"])
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    for ep in range(best["epochs"]):
        m.train()
        el = 0
        for b in tr_dl:
            opt.zero_grad()
            loss = crit(m(b["input_ids"].to(device), b["lengths"].to(device)), b["labels"].to(device))
            loss.backward(); opt.step(); el += loss.item()
        sched.step(el / len(tr_dl))

    m.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for b in te_dl:
            lg = m(b["input_ids"].to(device), b["lengths"].to(device))
            y_pred.extend((torch.softmax(lg, -1)[:, 1] >= THRESH_BI).long().cpu().tolist())
            y_true.extend(b["labels"].tolist())
    print("\n" + classification_report(y_true, y_pred, digits=3))
    return y_pred, y_true, cv, best


def _bert_cv(train_df, max_len, bs, lr, epochs, wd):
    tokenizer = load_tokenizer()
    X_all = train_df["text_clean"].tolist()
    y_all = train_df["label"].to_numpy()
    g_all = train_df["pair_id"].to_numpy()   # ← только train
    fold_f1s = []
    for fold, (tr_i, val_i) in enumerate(GroupKFold(5).split(np.zeros(len(y_all)), y_all, g_all), 1):
        tr_ds = BertDataset([X_all[i] for i in tr_i], y_all[tr_i], tokenizer, max_len)
        va_ds = BertDataset([X_all[i] for i in val_i], y_all[val_i], tokenizer, max_len)
        tr_dl, va_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True), DataLoader(va_ds, batch_size=bs)
        mdl = load_bert(device)
        opt = AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
        total_st = len(tr_dl) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(total_st * WARMUP_RATIO), total_st)
        for _ in range(epochs): train_one_epoch(mdl, tr_dl, opt, sched, device)
        metrics, _ = eval_bert(mdl, va_dl, device)
        fold_f1s.append(metrics["f1"])
        del mdl
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.mean(fold_f1s), np.std(fold_f1s)


def bert_hyperparam_search(train_df):
    configs = [
        {"max_len": 128, "bs": 16, "lr": 3e-5, "ep": 4, "wd": 0.01},
        {"max_len": 256, "bs": 16, "lr": 2e-5, "ep": 4, "wd": 0.01},
        {"max_len": 256, "bs": 16, "lr": 3e-5, "ep": 4, "wd": 0.01},
        {"max_len": 256, "bs": 16, "lr": 5e-5, "ep": 4, "wd": 0.01},
        {"max_len": 256, "bs": 8,  "lr": 3e-5, "ep": 4, "wd": 0.01},
        {"max_len": 256, "bs": 32, "lr": 3e-5, "ep": 4, "wd": 0.01},
        {"max_len": 256, "bs": 16, "lr": 3e-5, "ep": 3, "wd": 0.01},
        {"max_len": 256, "bs": 16, "lr": 3e-5, "ep": 5, "wd": 0.01},
        {"max_len": 384, "bs": 16, "lr": 3e-5, "ep": 4, "wd": 0.01},
    ]
    print("RuBERT Hyperparam Search")
    best_f1, best_cfg = 0, configs[2]
    for cfg in configs:
        mf, sf = _bert_cv(train_df, cfg["max_len"], cfg["bs"], cfg["lr"], cfg["ep"], cfg["wd"])  # ← только train
        print(f"  max_len={cfg['max_len']}, bs={cfg['bs']}, lr={cfg['lr']}, ep={cfg['ep']}: F1={mf:.4f}+-{sf:.4f}")
        if mf > best_f1: best_f1, best_cfg = mf, cfg
    print(f"Best: {best_cfg}, F1={best_f1:.4f}")
    return best_cfg


def bert_shap_analysis(model, tokenizer, test_df, n_samples=5):
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    tok_shap = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=False)
    masker = shap.maskers.Text(tok_shap, mask_token="[MASK]")
    def pred_fn(texts):
        model.eval(); res = []
        for t in texts:
            enc = tokenizer(str(t), return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
            with torch.no_grad(): logits = model(**enc).logits
            res.append(F.softmax(logits, -1).cpu().numpy()[0])
        return np.array(res)
    expl = shap.Explainer(pred_fn, masker, output_names=["pop", "sci"])
    sv = expl(test_df["text_clean"].iloc[:n_samples].tolist())
    print("\nSHAP RuBERT")
    for i in range(n_samples):
        print(f"\nSample {i}: {test_df['text_clean'].iloc[i][:60]}...")


def run_bert(data, train_df, test_df):
    best = bert_hyperparam_search(train_df)   # ← только train

    tokenizer = load_tokenizer()
    X_all = train_df["text_clean"].tolist()
    y_all = train_df["label"].to_numpy()
    g_all = train_df["pair_id"].to_numpy()   # ← только train
    folds = []
    for fold, (tr_i, val_i) in enumerate(GroupKFold(5).split(np.zeros(len(y_all)), y_all, g_all), 1):
        tr_ds = BertDataset([X_all[i] for i in tr_i], y_all[tr_i], tokenizer, best["max_len"])
        va_ds = BertDataset([X_all[i] for i in val_i], y_all[val_i], tokenizer, best["max_len"])
        tr_dl, va_dl = DataLoader(tr_ds, batch_size=best["bs"], shuffle=True), DataLoader(va_ds, batch_size=best["bs"])
        mdl = load_bert(device)
        opt = AdamW(mdl.parameters(), lr=best["lr"], weight_decay=best["wd"])
        tot = len(tr_dl) * best["ep"]
        sched = get_linear_schedule_with_warmup(opt, int(tot * WARMUP_RATIO), tot)
        for _ in range(best["ep"]): train_one_epoch(mdl, tr_dl, opt, sched, device)
        metrics, _ = eval_bert(mdl, va_dl, device)
        folds.append({"fold": fold, **metrics})
        print(f"  Fold {fold}: acc={metrics['acc']:.3f}, f1={metrics['f1']:.3f}")
        del mdl
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    cv = pd.DataFrame(folds)
    print(f"RuBERT: Acc={cv['acc'].mean():.3f}+-{cv['acc'].std():.3f}, F1={cv['f1'].mean():.3f}+-{cv['f1'].std():.3f}")

    tr_ds = BertDataset(train_df["text_clean"].tolist(), train_df["label"].tolist(), tokenizer, best["max_len"])
    te_ds = BertDataset(test_df["text_clean"].tolist(), test_df["label"].tolist(), tokenizer, best["max_len"])
    tr_dl, te_dl = DataLoader(tr_ds, batch_size=best["bs"], shuffle=True), DataLoader(te_ds, batch_size=best["bs"])
    bert_fin = load_bert(device)
    opt = AdamW(bert_fin.parameters(), lr=best["lr"], weight_decay=best["wd"])
    tot = len(tr_dl) * best["ep"]
    sch = get_linear_schedule_with_warmup(opt, int(tot * WARMUP_RATIO), tot)
    for _ in range(best["ep"]): train_one_epoch(bert_fin, tr_dl, opt, sch, device)
    m, y_pred = eval_bert(bert_fin, te_dl, device)
    y_true = test_df["label"].tolist()
    print("\n" + classification_report(y_true, y_pred, digits=3))

    bert_shap_analysis(bert_fin, tokenizer, test_df)
    del bert_fin
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return y_pred, y_true, cv, best


def multi_seed_classifiers(train_df, test_df, best_bi_cfg, best_bert_cfg):
    SEEDS = [random.randint(0, 2**31 - 1) for _ in range(3)]
    print("\nMULTI-SEED STABILITY (BiLSTM + RuBERT)")

    bi_seed_results = []
    print("\n--- BiLSTM ---")
    for s in SEEDS:
        set_seed(s)
        vocab = build_char_vocab(train_df["text"].tolist())
        tr_dl = DataLoader(CharDataset(train_df, vocab), batch_size=BATCH_SIZE, shuffle=True)
        te_dl = DataLoader(CharDataset(test_df, vocab),  batch_size=BATCH_SIZE)
        m = CharCNNBiLSTM(len(vocab), best_bi_cfg["emb"], best_bi_cfg["hid"], best_bi_cfg["drop"]).to(device)
        opt = AdamW(m.parameters(), lr=best_bi_cfg["lr"])
        crit = nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        for _ in range(best_bi_cfg.get("epochs", EPOCHS_BI)):
            m.train(); el = 0
            for b in tr_dl:
                opt.zero_grad()
                loss = crit(m(b["input_ids"].to(device), b["lengths"].to(device)), b["labels"].to(device))
                loss.backward(); opt.step(); el += loss.item()
            sched.step(el / len(tr_dl))
        m.eval(); y_pred, y_true = [], []
        with torch.no_grad():
            for b in te_dl:
                lg = m(b["input_ids"].to(device), b["lengths"].to(device))
                y_pred.extend((torch.softmax(lg, -1)[:, 1] >= THRESH_BI).long().cpu().tolist())
                y_true.extend(b["labels"].tolist())
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        print(f"  seed={s}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        bi_seed_results.append({"seed": s, "acc": acc, "prec": prec, "rec": rec, "f1": f1})
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    bert_seed_results = []
    print("\n--- RuBERT ---")
    tokenizer = load_tokenizer()
    for s in SEEDS:
        set_seed(s)
        tr_ds = BertDataset(train_df["text_clean"].tolist(), train_df["label"].tolist(),
                             tokenizer, best_bert_cfg["max_len"])
        te_ds = BertDataset(test_df["text_clean"].tolist(),  test_df["label"].tolist(),
                             tokenizer, best_bert_cfg["max_len"])
        tr_dl = DataLoader(tr_ds, batch_size=best_bert_cfg["bs"], shuffle=True)
        te_dl = DataLoader(te_ds, batch_size=best_bert_cfg["bs"])
        bert_fin = load_bert(device)
        opt = AdamW(bert_fin.parameters(), lr=best_bert_cfg["lr"],
                    weight_decay=best_bert_cfg["wd"])
        tot = len(tr_dl) * best_bert_cfg["ep"]
        sch = get_linear_schedule_with_warmup(opt, int(tot * WARMUP_RATIO), tot)
        for _ in range(best_bert_cfg["ep"]):
            train_one_epoch(bert_fin, tr_dl, opt, sch, device)
        metrics, _ = eval_bert(bert_fin, te_dl, device)
        print(f"  seed={s}: Acc={metrics['acc']:.4f}, Prec={metrics['prec']:.4f}, "
              f"Rec={metrics['rec']:.4f}, F1={metrics['f1']:.4f}")
        bert_seed_results.append({"seed": s, **metrics})
        del bert_fin
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\nMulti-seed summary (mean ± std по 3 сидам, тестовая выборка)")
    bi = pd.DataFrame(bi_seed_results)
    bert = pd.DataFrame(bert_seed_results)
    print(f"  BiLSTM : Acc={bi['acc'].mean():.4f}+-{bi['acc'].std():.4f}, "
          f"F1={bi['f1'].mean():.4f}+-{bi['f1'].std():.4f}")
    print(f"  RuBERT : Acc={bert['acc'].mean():.4f}+-{bert['acc'].std():.4f}, "
          f"F1={bert['f1'].mean():.4f}+-{bert['f1'].std():.4f}")
    return bi, bert


def main():
    data = load_data()
    train_df, test_df = train_test_by_pairs(data)
    print(f"train: {len(train_df)}, test: {len(test_df)}")
    assert len(set(train_df["pair_id"]) & set(test_df["pair_id"])) == 0

    print("\n1. BASELINE")
    y_bl, bl_cv = run_baseline(data, train_df, test_df)

    print("\n2. SVM")
    y_svm, svm_cv = run_svm(data, train_df, test_df)

    print("\n3. Char-CNN-BiLSTM")
    y_bi, yt_bi, bi_cv, best_bi_cfg = run_bilstm(data, train_df, test_df)

    print("\n4. RuBERT")
    y_bert, yt_bert, bert_cv, best_bert_cfg = run_bert(data, train_df, test_df)

    multi_seed_classifiers(train_df, test_df, best_bi_cfg, best_bert_cfg)

    print("\nITOGI")
    summary = {
        "Baseline": clf_metrics(test_df["label"].values, y_bl),
        "SVM": clf_metrics(test_df["label"].values, y_svm),
        "BiLSTM": clf_metrics(yt_bi, y_bi),
        "RuBERT": clf_metrics(yt_bert, y_bert),
    }
    print(pd.DataFrame(summary).T.round(3))

    print("\nCV F1 summary")
    for name, cv in [("Baseline", bl_cv), ("SVM", svm_cv), ("BiLSTM", bi_cv), ("RuBERT", bert_cv)]:
        print(f"  {name:<15} F1={cv['f1'].mean():.4f}+-{cv['f1'].std():.4f}")


if __name__ == "__main__":
    main()
