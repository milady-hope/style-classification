from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.config import (
    SVM_C, SVM_ANALYZER, SVM_NGRAM, SVM_MIN_DF, SVM_MAX_DF,
    SVM_SUBLINEAR_TF,
)


def build_svm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=SVM_ANALYZER,
            ngram_range=SVM_NGRAM,
            min_df=SVM_MIN_DF,
            max_df=SVM_MAX_DF,
            sublinear_tf=SVM_SUBLINEAR_TF,
            norm="l2",
        )),
        ("clf", LinearSVC(C=SVM_C, max_iter=1000)),
    ])
