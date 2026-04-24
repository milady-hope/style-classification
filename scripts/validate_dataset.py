import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from src.preprocessing import load_data
from src.features import extract_one, FEAT_NAMES

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+")


def main():
    df = load_data()
    print(f"Строк: {len(df)}, пар: {df['pair_id'].nunique()}")

    print("\nПропуски:", df.isnull().sum().to_dict())
    empty = df[df["text"].str.len() == 0]
    print(f"Пустых текстов: {len(empty)}")

    pair_check = df.groupby("pair_id").agg(
        n=("text", "count"),
        labels=("label", lambda x: sorted(x.tolist())),
    )
    bad = pair_check[
        (pair_check["n"] != 2) | (pair_check["labels"].apply(lambda x: x != [0, 1]))
    ]
    print(f"Пар с нарушенной структурой: {len(bad)}")

    counts = df["label"].value_counts().sort_index()
    print(f"\nБаланс: {counts.to_dict()}")

    df["n_words"] = df["text"].apply(lambda t: len(TOKEN_RE.findall(t)))
    df["n_sents"] = df["text"].apply(
        lambda t: max(len([s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]), 1)
    )

    pop = df[df["label"] == 0]
    sci = df[df["label"] == 1]
    print(f"\nСр. длина (слова): науч.-поп.={pop['n_words'].mean():.1f}, "
          f"научный={sci['n_words'].mean():.1f}")

    U, p = stats.mannwhitneyu(pop["n_words"], sci["n_words"], alternative="two-sided")
    print(f"Mann-Whitney U: U={U:.0f}, p={p:.2e}")

    feat_matrix = np.vstack([extract_one(t) for t in df["text"].values])
    markers_df = pd.DataFrame(feat_matrix, columns=FEAT_NAMES, index=df.index)
    df = pd.concat([df, markers_df], axis=1)

    print(f"\n{'Маркер':<20} {'U':>10} {'p-value':>12}")
    for col in FEAT_NAMES:
        v0 = df.loc[df["label"] == 0, col].values
        v1 = df.loc[df["label"] == 1, col].values
        U, p = stats.mannwhitneyu(v0, v1, alternative="two-sided")
        print(f"{col:<20} {U:>10.0f} {p:>12.2e}")

    n_dup = df.duplicated(subset=["text"], keep=False).sum()
    print(f"\nТочных дубликатов: {n_dup}")

    def jaccard(t1, t2):
        w1 = set(TOKEN_RE.findall(str(t1).lower()))
        w2 = set(TOKEN_RE.findall(str(t2).lower()))
        return len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0

    pairs = df.pivot_table(index="pair_id", columns="label", values="text", aggfunc="first")
    pairs.columns = ["pop", "sci"]
    pairs["jaccard"] = pairs.apply(lambda r: jaccard(r["pop"], r["sci"]), axis=1)
    print(f"\nJaccard внутри пар: mean={pairs['jaccard'].mean():.3f}, "
          f"median={pairs['jaccard'].median():.3f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random.randint(0, 2**31 - 1))
    tr_idx, te_idx = next(gss.split(df["text"], df["label"], groups=df["pair_id"]))
    tr_pairs = set(df.iloc[tr_idx]["pair_id"])
    te_pairs = set(df.iloc[te_idx]["pair_id"])
    print(f"\nTrain: {len(tr_idx)}, Test: {len(te_idx)}")
    print(f"Пересечение пар: {len(tr_pairs & te_pairs)}")

    gkf = GroupKFold(n_splits=5)
    y = df["label"].values
    groups = df["pair_id"].values
    print("\nGroupKFold (5 фолдов):")
    for fold, (tr, val) in enumerate(gkf.split(np.zeros(len(y)), y, groups), 1):
        overlap = len(set(groups[tr]) & set(groups[val]))
        print(f"  Fold {fold}: train={len(tr)}, val={len(val)}, overlap={overlap}")


if __name__ == "__main__":
    main()
