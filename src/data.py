"""Nạp dữ liệu UIT-VSFC (Vietnamese Students' Feedback Corpus) hoặc CSV tự có.

Nhãn cảm xúc: 0 = tiêu cực, 1 = trung tính, 2 = tích cực.
Nguồn mặc định: Hugging Face `uitnlp/vietnamese_students_feedback` (cột `sentence`, `sentiment`).
Hãy kiểm tra tên bộ dữ liệu, tên cột và giấy phép trước khi dùng.
Dùng CSV riêng: --csv_dir thư mục có train.csv / validation.csv / test.csv với 2 cột `text`, `label`.
"""
from __future__ import annotations
import os
import pandas as pd

LABELS = ["tiêu cực", "trung tính", "tích cực"]
SPLITS = ("train", "validation", "test")


def load_splits(csv_dir: str | None = None, hf_name: str = "uitnlp/vietnamese_students_feedback") -> dict[str, pd.DataFrame]:
    if csv_dir:
        return {s: pd.read_csv(os.path.join(csv_dir, f"{s}.csv"))[["text", "label"]].dropna() for s in SPLITS}
    from datasets import load_dataset
    ds = load_dataset(hf_name)
    out = {}
    for s in SPLITS:
        df = ds[s].to_pandas().rename(columns={"sentence": "text", "sentiment": "label"})
        out[s] = df[["text", "label"]].dropna().reset_index(drop=True)
    return out


def describe(splits: dict[str, pd.DataFrame]) -> str:
    lines = []
    for s, df in splits.items():
        dist = df["label"].value_counts(normalize=True).sort_index().round(3).to_dict()
        lines.append(f"{s}: {len(df)} câu, phân bố nhãn {dist}")
    return "\n".join(lines)
