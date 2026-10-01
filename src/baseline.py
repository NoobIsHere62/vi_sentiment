"""Mô hình nền: TF-IDF (từ + ký tự) + Logistic Regression. Chạy được trên laptop.

    python src/baseline.py                 # tải UIT-VSFC từ Hugging Face
    python src/baseline.py --csv_dir data  # hoặc dùng CSV riêng
Chọn siêu tham số C bằng tập validation, báo cáo cuối cùng trên tập test (chỉ chạy test một lần).
"""
from __future__ import annotations
import argparse
import os
import sys
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import describe, load_splits
from evaluate import compare, evaluate


def features(train_texts):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=True)
    word.fit(train_texts); char.fit(train_texts)
    return word, char


def transform(word, char, texts):
    return sp.hstack([word.transform(texts), char.transform(texts)]).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model_dir", default="models/baseline")
    a = ap.parse_args()

    d = load_splits(a.csv_dir)
    print(describe(d))
    word, char = features(d["train"]["text"])
    Xtr, Xva, Xte = (transform(word, char, d[s]["text"]) for s in ("train", "validation", "test"))

    best = None
    for C in (0.3, 1, 3, 10, 30):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, d["train"]["label"])
        f1 = f1_score(d["validation"]["label"], clf.predict(Xva), average="macro")
        print(f"C={C:<5} validation macro-F1 = {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    print(f"-> chọn C={C}")

    res = evaluate("tfidf_logreg", d["test"]["text"], d["test"]["label"], clf.predict(Xte), a.out)
    print(res)
    os.makedirs(a.model_dir, exist_ok=True)
    joblib.dump({"word": word, "char": char, "clf": clf}, os.path.join(a.model_dir, "model.joblib"))
    print(compare(a.out))


if __name__ == "__main__":
    main()
