"""Demo web (Flask): nhập một câu phản hồi → dự đoán cảm xúc + độ tin cậy.

Dùng mô hình PhoBERT nếu có thư mục models/phobert (và đã cài torch/transformers/pyvi), ngược lại dùng mô hình nền TF-IDF.
    python app.py        # http://127.0.0.1:5000
"""
from __future__ import annotations
import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data import LABELS

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
MODEL = {"name": None, "predict": None}


def load_model():
    pdir = os.path.join(ROOT, "models", "phobert")
    if os.path.isdir(pdir):
        try:
            import torch
            from pyvi import ViTokenizer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(pdir)
            mdl = AutoModelForSequenceClassification.from_pretrained(pdir).eval()

            def predict(text):
                enc = tok(ViTokenizer.tokenize(text), return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    return torch.softmax(mdl(**enc).logits, -1)[0].tolist()
            MODEL.update(name="PhoBERT", predict=predict)
            return
        except Exception as e:                                   # thiếu thư viện → dùng mô hình nền
            print(f"[demo] không nạp được PhoBERT ({type(e).__name__}); dùng TF-IDF.")
    import joblib
    import scipy.sparse as sp
    m = joblib.load(os.path.join(ROOT, "models", "baseline", "model.joblib"))

    def predict(text):
        X = sp.hstack([m["word"].transform([text]), m["char"].transform([text])]).tocsr()
        return m["clf"].predict_proba(X)[0].tolist()
    MODEL.update(name="TF-IDF + Logistic Regression", predict=predict)


@app.route("/")
def index():
    return render_template("index.html", model=MODEL["name"])


@app.route("/api/predict", methods=["POST"])
def api():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()[:500]
    if not text:
        return jsonify(error="Hãy nhập một câu."), 400
    p = MODEL["predict"](text)
    k = max(range(3), key=lambda i: p[i])
    return jsonify(label=LABELS[k], confidence=round(p[k], 3), probs={LABELS[i]: round(p[i], 3) for i in range(3)})


load_model()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
