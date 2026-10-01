# Vietnamese Sentiment Classification — TF-IDF baseline vs. PhoBERT

Phân loại cảm xúc (tiêu cực / trung tính / tích cực) cho phản hồi tiếng Việt. So sánh một mô hình nền cổ điển
(TF-IDF + Logistic Regression) với **PhoBERT** được fine-tune, đánh giá trên cùng tập test, có phân tích lỗi và demo web.

> Trạng thái: mã nguồn đã sẵn sàng. **Kết quả thật** nằm ở mục [Kết quả](#kết-quả), được điền sau khi chạy trên bộ dữ liệu thật.

## Quy trình

1. **Dữ liệu:** UIT-VSFC (Vietnamese Students' Feedback Corpus) qua Hugging Face `uitnlp/vietnamese_students_feedback`, hoặc CSV riêng (`text`, `label`). Cần kiểm tra giấy phép và tên bộ dữ liệu trước khi dùng.
2. **Mô hình nền:** TF-IDF (từ 1–2-gram + ký tự 2–5-gram) + Logistic Regression cân bằng lớp; chọn `C` trên tập validation.
3. **PhoBERT:** `vinai/phobert-base-v2`, tách từ bằng `pyvi`, fine-tune với AdamW, warmup, chọn checkpoint tốt nhất theo macro-F1 trên validation.
4. **Đánh giá:** accuracy, macro-F1, F1 từng lớp, ma trận nhầm lẫn, danh sách câu đoán sai (`results/*_errors.csv`). Tập test chỉ dùng một lần ở cuối.
5. **Demo:** Flask, dùng PhoBERT nếu có `models/phobert`, ngược lại dùng mô hình nền.

## Cấu trúc

```
├── src/
│   ├── data.py               nạp dữ liệu, nhãn
│   ├── evaluate.py           chỉ số, ma trận nhầm lẫn, phân tích lỗi, bảng so sánh
│   ├── baseline.py           TF-IDF + Logistic Regression
│   └── finetune_phobert.py   fine-tune PhoBERT (chạy trên Colab GPU)
├── app.py, templates/        demo web
├── colab_train.ipynb         chạy toàn bộ trên Google Colab
├── results/                  metrics, ma trận nhầm lẫn, câu sai, comparison.md
└── models/                   mô hình đã huấn luyện (không commit)
```

## Chạy

```bash
pip install -r requirements.txt
python src/baseline.py                  # mô hình nền, chạy được trên laptop
python app.py                           # demo tại http://127.0.0.1:5000
```

Fine-tune PhoBERT: mở `colab_train.ipynb` trên Google Colab (GPU T4), chạy lần lượt các ô, tải `results_and_model.zip` về
rồi giải nén vào thư mục dự án để demo dùng PhoBERT.

## Kết quả

*(Điền sau khi chạy thật — dán nội dung `results/comparison.md` vào đây.)*

| Mô hình | Accuracy | Macro-F1 | F1 (tiêu cực / trung tính / tích cực) |
|---|---|---|---|
| TF-IDF + Logistic Regression | … | … | … |
| PhoBERT (fine-tuned) | … | … | … |

## Phân tích lỗi

*(Điền sau: các kiểu câu bị đoán sai nhiều nhất, ví dụ câu trung tính bị nhầm sang tích cực, câu có phủ định hoặc mỉa mai.)*

## Hạn chế

- Kết quả phụ thuộc bộ dữ liệu (phản hồi sinh viên) nên chưa chắc đúng cho miền khác như đánh giá sản phẩm hay mạng xã hội.
- Chỉ một lần chạy với một hạt giống ngẫu nhiên; chênh lệch nhỏ giữa hai mô hình có thể nằm trong dao động ngẫu nhiên.

## Giấy phép

[MIT](LICENSE) © Nguyễn Hoàng Tuấn
