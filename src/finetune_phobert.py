"""Fine-tune PhoBERT cho phân loại cảm xúc. Nên chạy trên Google Colab (GPU T4 miễn phí).

    !python src/finetune_phobert.py --epochs 4
PhoBERT cần văn bản đã tách từ (word segmentation); ở đây dùng thư viện pyvi.
Kết quả: results/phobert_metrics.json, ma trận nhầm lẫn, danh sách câu sai; model lưu ở models/phobert/.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
from pyvi import ViTokenizer
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

sys.path.insert(0, os.path.dirname(__file__))
from data import LABELS, describe, load_splits
from evaluate import compare, evaluate


def seg(t: str) -> str:
    return ViTokenizer.tokenize(str(t))


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok([seg(t) for t in texts], truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=None)
    ap.add_argument("--model", default="vinai/phobert-base-v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model_dir", default="models/phobert")
    a = ap.parse_args()
    set_seed(a.seed)

    d = load_splits(a.csv_dir)
    print(describe(d))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3, id2label=dict(enumerate(LABELS)))
    tr, va, te = (DS(d[s]["text"], d[s]["label"], tok, a.max_len) for s in ("train", "validation", "test"))

    def metrics(p):
        return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

    args = TrainingArguments(output_dir="checkpoints", num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=a.batch, per_device_eval_batch_size=64, weight_decay=0.01,
                             warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="macro_f1", save_total_limit=1, fp16=torch.cuda.is_available(),
                             report_to="none", seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metrics)
    trainer.train()

    pred = trainer.predict(te).predictions.argmax(-1)
    print(evaluate("phobert", d["test"]["text"], d["test"]["label"], pred, a.out))
    trainer.save_model(a.model_dir); tok.save_pretrained(a.model_dir)
    print(compare(a.out))


if __name__ == "__main__":
    main()
