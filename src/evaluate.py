"""Chỉ số đánh giá dùng chung cho mọi mô hình: accuracy, macro-F1, F1 từng lớp, ma trận nhầm lẫn, danh sách câu đoán sai."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import LABELS


def evaluate(name: str, texts, y_true, y_pred, out_dir: str = "results") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0)
    res = {"model": name, "n": int(len(y_true)),
           "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
           "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
           "f1_per_class": {k: round(rep[k]["f1-score"], 4) for k in LABELS}}
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(f"{name}  (macro-F1 = {res['macro_f1']})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{name}_confusion.png"), dpi=140); plt.close(fig)

    wrong = pd.DataFrame({"text": list(texts), "true": [LABELS[i] for i in y_true], "pred": [LABELS[i] for i in y_pred]})
    wrong[y_true != y_pred].to_csv(os.path.join(out_dir, f"{name}_errors.csv"), index=False, encoding="utf-8-sig")
    return res


def compare(out_dir: str = "results") -> str:
    """Bảng so sánh các mô hình đã có kết quả (đọc từ *_metrics.json)."""
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith("_metrics.json"):
            r = json.load(open(os.path.join(out_dir, fn), encoding="utf-8"))
            rows.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | " +
                        " / ".join(f"{r['f1_per_class'][k]:.3f}" for k in LABELS) + " |")
    head = "| Mô hình | Accuracy | Macro-F1 | F1 (tiêu cực / trung tính / tích cực) |\n|---|---|---|---|\n"
    table = head + "\n".join(rows)
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    return table
