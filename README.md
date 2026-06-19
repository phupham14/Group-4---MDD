# MDD với Wav2Vec2 — Mispronunciation Detection & Diagnosis

Hệ thống phát hiện và chẩn đoán lỗi phát âm (Mispronunciation Detection & Diagnosis) dựa trên mô hình **Wav2Vec2** tiếng Việt, được fine-tune bằng CTC loss trên tập dữ liệu MDD Challenge 2025.

---

## Mục lục

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt](#cài-đặt)
3. [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
4. [Cách thuật toán hoạt động](#cách-thuật-toán-hoạt-động)
   - [Bước 1 — Tiền xử lý dữ liệu](#bước-1--tiền-xử-lý-dữ-liệu)
   - [Bước 2 — Encode âm thanh (Feature Extraction)](#bước-2--encode-âm-thanh-feature-extraction)
   - [Bước 3 — Xây dựng Vocabulary & Tokenizer](#bước-3--xây-dựng-vocabulary--tokenizer)
   - [Bước 4 — Dataset & Oversampling](#bước-4--dataset--oversampling)
   - [Bước 5 — Mô hình Wav2Vec2ForCTC](#bước-5--mô-hình-wav2vec2forctc)
   - [Bước 6 — Training](#bước-6--training)
   - [Bước 7 — Decode (CTC Decoding)](#bước-7--decode-ctc-decoding)
   - [Bước 8 — MDD Diagnosis](#bước-8--mdd-diagnosis)
5. [Đánh giá](#đánh-giá)
6. [Inference](#inference)

---

## Yêu cầu hệ thống

- Python 3.9+
- GPU (khuyến nghị NVIDIA T4 trở lên, có thể chạy trên Google Colab)
- CUDA 11.x+ (nếu dùng GPU local)

---

## Cài đặt

```bash
pip install torch torchaudio
pip install transformers datasets
pip install librosa soundfile
pip install scikit-learn pandas numpy matplotlib seaborn
pip install editdistance tqdm
```

> Nếu chạy trên **Google Colab**, mount Google Drive trước:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> ```

---

## Cấu trúc dữ liệu

```
MDD-Challenge-2025-training-set/
├── audio_data/
│   └── train/          # Các file .wav của learner
├── metadata/
│   ├── train.csv       # id, path, transcript
│   └── train_phones.csv # id, canonical (chuỗi phoneme chuẩn), actual (chuỗi phoneme thực tế)
```

| Cột | Ý nghĩa |
|-----|---------|
| `transcript` | Văn bản của câu nói |
| `canonical` | Chuỗi phoneme chuẩn (giáo viên đọc đúng) |
| `actual` | Chuỗi phoneme người học thực sự phát âm |
| `has_error` | `1` nếu `canonical ≠ actual`, `0` nếu đúng |

---

## Cách thuật toán hoạt động

### Bước 1 — Tiền xử lý dữ liệu

**Normalize phoneme:** Loại bỏ số chỉ tông ở cuối mỗi phoneme (VD: `b-1` → `b`):

```python
def normalize_phoneme(p_string):
    return ' '.join([re.sub(r'-\d+', '', p) for p in p_string.split()])
```

**Gán nhãn lỗi:** So sánh chuỗi `target_norm` (canonical) với `actual_norm` để tạo nhãn nhị phân `has_error`.

**Stratified Split:** Chia dữ liệu thành 3 tập, giữ nguyên tỉ lệ nhãn `has_error`:

```
Toàn bộ dữ liệu
     │
     ├── 80% ─── Train
     └── 20% ─── Temp
                  ├── 50% ─── Validation  (= 10% tổng)
                  └── 50% ─── Test        (= 10% tổng)
```

---

### Bước 2 — Encode âm thanh (Feature Extraction)

Mỗi file `.wav` được tải ở **16 kHz** bằng `librosa`, sau đó đưa qua `Wav2Vec2FeatureExtractor`:

```
Audio (.wav 16kHz)
        │
        ▼
Wav2Vec2FeatureExtractor
  - Chuẩn hóa biên độ (zero-mean, unit-variance)
  - Trả về tensor input_values (shape: [T])
        │
        ▼
Wav2Vec2 Feature Encoder (CNN stack)
  - 7 lớp Conv1D với stride giảm dần
  - Nén từ 16000 samples/giây → ~50 frames/giây
  - Output: sequence of feature vectors (shape: [T', 512])
        │
        ▼
Wav2Vec2 Transformer Encoder
  - 12 lớp Transformer với relative positional encoding
  - Học biểu diễn ngữ cảnh của âm thanh
  - Output: context vectors (shape: [T', 768])
```

> Pretrained checkpoint: [`nguyenvulebinh/wav2vec2-base-vietnamese-250h`](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) — được pre-train trên 250 giờ tiếng Việt.

---

### Bước 3 — Xây dựng Vocabulary & Tokenizer

Vocabulary được xây dựng từ **toàn bộ phoneme** xuất hiện trong tập train (cả `target_norm` và `actual_norm`):

```python
vocab = {phone: idx for idx, phone in enumerate(unique_phones)}
# Thêm các token đặc biệt:
# [PAD] → CTC blank token
# [UNK] → phoneme chưa gặp
```

Sau đó lưu thành `vocab.json` để khởi tạo `Wav2Vec2CTCTokenizer`:

```python
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]")
processor  = Wav2Vec2Processor(feature_extractor=..., tokenizer=tokenizer)
```

---

### Bước 4 — Dataset & Oversampling

**MDDDataset** mỗi lần `__getitem__` sẽ:
1. Load file `.wav` → tensor `input_values`
2. Map chuỗi phoneme `actual_norm` → danh sách token ID làm `labels`

**Vấn đề mất cân bằng:** Chỉ ~6.7% mẫu có lỗi phát âm.  
**Giải pháp:** Oversampling — nhân bản các mẫu lỗi cho đến khi tỉ lệ lỗi đạt **50%** trong tập train:

```
Trước oversampling:  93.3% đúng / 6.7% lỗi
Sau oversampling:    50%   đúng / 50%  lỗi
```

**Collate function** xử lý batch:
- Padding `input_values` về cùng độ dài (pad bằng `0.0`)
- Tạo `attention_mask` để model bỏ qua phần padding
- Padding `labels` bằng `-100` (PyTorch bỏ qua khi tính loss)

---

### Bước 5 — Mô hình Wav2Vec2ForCTC

```
Input audio
     │
     ▼
[Feature Encoder CNN] ──── frozen (không update trọng số)
     │
     ▼
[Transformer Encoder]
     │
     ▼
[CTC Linear Head]  vocab_size = số phoneme
     │
     ▼
Logits (shape: [T', vocab_size])
```

- **Feature Encoder bị đóng băng** (`model.freeze_feature_encoder()`) — giữ nguyên khả năng trích xuất đặc trưng âm thanh cấp thấp đã học từ pre-training.
- **Transformer Encoder được fine-tune** để học biểu diễn phoneme cho tiếng Việt L2 (người học).
- **CTC Head** là một lớp Linear mới với `vocab_size` khớp với vocabulary tự xây dựng.

---

### Bước 6 — Training

```python
TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,   # effective batch = 16
    learning_rate               = 3e-5,
    num_train_epochs            = 30,
    fp16                        = True,  # Mixed precision trên GPU
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss"
)
```

**CTC Loss** được tính giữa logits của model và chuỗi phoneme `actual` (những gì người học thực sự nói), không phải `canonical`.

---

### Bước 7 — Decode (CTC Decoding)

Sau khi model cho ra `logits`, ta lấy `argmax` theo chiều vocab để ra chuỗi token ID thô, rồi áp dụng **CTC decode**:

```
Logits [T', vocab_size]
        │ argmax
        ▼
Token ID sequence (raw):  [3, 3, 0, 5, 5, 5, 0, 2, 2]
        │ CTC decode
        │  1. Bỏ blank token (id = [PAD])
        │  2. Gộp các token lặp liên tiếp
        ▼
Decoded phoneme sequence: ["b", "aa", "n"]
```

Ví dụ minh họa:

```
Raw:      [b, b, PAD, aa, aa, aa, PAD, n, n]
Bỏ blank: [b, b,      aa, aa, aa,      n, n]
Gộp lặp:  [b,         aa,              n   ]
Kết quả:  ["b", "aa", "n"]
```

---

### Bước 8 — MDD Diagnosis

So sánh chuỗi **canonical** (đúng) với chuỗi **predicted** (model dự đoán) bằng **Dynamic Programming (Edit Distance)**:

```
canonical:  ["b", "aa", "n",  "d"]
predicted:  ["b", "oo", "n"      ]
                   │              │
              substitution    deletion
```

Thuật toán truy vết ngược bảng DP để phân loại từng phoneme:

| Loại | Ý nghĩa | Ví dụ |
|------|---------|-------|
| `correct` | Phát âm đúng | canonical=`b`, predicted=`b` |
| `substitution` | Phát âm sai phoneme | canonical=`aa`, predicted=`oo` |
| `deletion` | Bỏ sót phoneme | canonical=`d`, predicted=`-` |
| `insertion` | Thêm phoneme thừa | canonical=`-`, predicted=`k` |

---

## Đánh giá

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **PER** | (S+D+I) / N | Phoneme Error Rate |
| **F1** | F1(y_true, y_pred) | Phát hiện câu có lỗi |
| **DER** | errors / total | Chẩn đoán đúng loại lỗi |
| **Final Score** | `0.5×F1 + 0.4×(1−DER) + 0.1×(1−PER)` | Điểm tổng hợp chính thức |

---

## Inference

```python
import librosa, torch

# Load audio
speech, _ = librosa.load("path/to/audio.wav", sr=16000)

# Feature extraction
inputs = processor(speech, sampling_rate=16000, return_tensors="pt")

# Model inference
vi_model.eval()
with torch.no_grad():
    logits = vi_model(input_values=inputs.input_values).logits

# CTC Decode
pred_ids   = torch.argmax(logits, dim=-1)[0]
predicted  = ctc_decode(pred_ids, id2phone, vocab['[PAD]'])

# MDD Diagnosis
canonical  = "b aa n d".split()   # chuỗi phoneme chuẩn
diagnosis  = get_mdd_diagnosis(canonical, predicted)

import pandas as pd
print(pd.DataFrame(diagnosis))
```

**Kết quả mẫu:**

```
  canonical  pred   diagnosis
0         b     b     correct
1        aa    oo substitution
2         n     n     correct
3         d     -    deletion
```
