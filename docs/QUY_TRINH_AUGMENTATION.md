# Quy trình Data Augmentation

Tài liệu mô tả **quy trình** áp dụng Data Augmentation cho dataset PPE (mũ/áo bảo hộ) trước khi train YOLO, giúp mô hình thích nghi tốt hơn với nhiều điều kiện môi trường (ánh sáng, thời tiết, góc máy, che khuất).

---

## 1. Tổng quan quy trình

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Dataset gốc    │     │  Augmentation        │     │  Dataset đã     │
│  (YOLO format)  │ ──► │  Pipeline            │ ──► │  augment        │
│  train/         │     │  (ảnh + transform     │     │  train_aug/     │
│  val/           │     │   bounding box)      │     │  (hoặc gộp)    │
└─────────────────┘     └──────────────────────┘     └────────┬────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────────┐
                                                      │  Train YOLO     │
                                                      │  (data.yaml     │
                                                      │   trỏ train_aug)│
                                                      └─────────────────┘
```

- **Đầu vào:** Thư mục ảnh + file label YOLO (mỗi dòng: `class_id x_center y_center width height`, normalized 0–1).
- **Xử lý:** Áp dụng pipeline augmentation (ánh sáng, màu, nhiễu, hình học, che khuất); mỗi transform có xác suất `p`; bounding box được cập nhật đúng khi có biến đổi hình học.
- **Đầu ra:** Thư mục ảnh mới + label mới (format YOLO giữ nguyên), dùng làm dữ liệu train (thay hoặc bổ sung cho dataset gốc).

---

## 2. Chuẩn bị dữ liệu đầu vào

### 2.1. Cấu trúc thư mục (YOLO chuẩn)

```
dataset/
├── train/
│   ├── images/     # Ảnh train (jpg, png, ...)
│   └── labels/     # File .txt cùng tên với ảnh
└── val/
    ├── images/
    └── labels/
```

- Mỗi ảnh `images/xxx.jpg` có file label tương ứng `labels/xxx.txt`.
- **Chỉ augment tập train.** Tập `val` giữ nguyên để đánh giá khách quan.

### 2.2. Định dạng file label

Mỗi dòng trong file `.txt`:

```
class_id x_center y_center width height
```

- Tất cả giá trị **normalized** trong đoạn [0, 1] (chia cho kích thước ảnh).
- Ví dụ: `2 0.45 0.52 0.18 0.31` — class 2 (Non-Helmet), tâm (45% chiều rộng, 52% chiều cao), rộng 18%, cao 31%.

Thứ tự class phải khớp với `data.yaml` khi train (ví dụ: 0 Gloves, 1 Helmet, 2 Non-Helmet, 3 Person, 4 Shoes, 5 Vest, 6 bare-arms).

---

## 3. Các bước trong pipeline augmentation

Pipeline mặc định (`AugmentPipeline.default_training()`) áp dụng **theo thứ tự** các transform dưới đây. Mỗi transform có xác suất `p` (chỉ áp dụng khi random &lt; p).

### Bước 3.1. Ánh sáng và màu (không đổi tọa độ box)

| Transform | Tham số chính | Mục đích |
|-----------|----------------|----------|
| **RandomBrightness** | `delta` (độ lệch sáng) | Nắng gắt, bóng râm, đèn yếu |
| **RandomContrast** | `range_contrast` | Màn hình/ánh sáng khác nhau |
| **RandomGamma** | `range_gamma` | Low-light, overexposed |
| **RandomHSV** | `h`, `s`, `v` | Đèn vàng, bụi bẩn màu, thời tiết |

→ Mô phỏng **điều kiện chụp khác nhau** trên công trường.

### Bước 3.2. Nhiễu và mờ (không đổi box)

| Transform | Tham số chính | Mục đích |
|-----------|----------------|----------|
| **GaussianNoise** | `var_limit` | Nhiễu sensor, ánh sáng kém |
| **GaussianBlur** | `kernel_range` | Sương, kính bẩn, out-of-focus |
| **MotionBlur** | `max_kernel` | Camera/người chuyển động |

→ Mô phỏng **chất lượng ảnh** và **môi trường** (bụi, ẩm).

### Bước 3.3. Hình học (có transform bounding box)

| Transform | Tham số chính | Mục đích |
|-----------|----------------|----------|
| **HorizontalFlip** | `p` | Góc quay camera trái/phải |
| **VerticalFlip** | `p` | Góc từ trên/xuống (ít dùng) |
| **RandomRotate** | `angle_range` | Camera nghiêng, góc chụp khác |
| **RandomScaleCrop** | `scale_range` | Camera xa/gần, crop ngẫu nhiên |
| **RandomPerspective** | `distort_scale` | Góc xiên, gần thực tế công trường |

→ Mô hình học được **nhiều góc máy** và **khoảng cách** khác nhau.

### Bước 3.4. Che khuất (không đổi box)

| Transform | Tham số chính | Mục đích |
|-----------|----------------|----------|
| **RandomCutout** | `n_holes`, `size_ratio` | Vật che, bụi, lỗi vùng ảnh |

→ Tăng **robust** khi đối tượng bị che một phần.

---

## 4. Chạy augmentation (quy trình thực hiện)

### 4.1. Trên máy local (từ thư mục gốc dự án)

```bash
python -m src.training.prepare_augmented_dataset \
  --images data/train/images \
  --labels data/train/labels \
  --out data/train_aug \
  --num 2 \
  --copy-original
```

- **`--images`**, **`--labels`**: thư mục ảnh và label **train** (YOLO format).
- **`--out`**: thư mục đích; script tạo `out/images/` và `out/labels/`.
- **`--num 2`**: mỗi ảnh gốc sinh thêm **2** bản augment (tên dạng `xxx_aug0`, `xxx_aug1`).
- **`--copy-original`**: copy cả ảnh và label **gốc** vào `out` (nên bật để vừa có gốc vừa có augment).

Kết quả: dataset mở rộng trong `data/train_aug/`, sẵn sàng dùng cho bước train.

### 4.2. Trên Google Colab

- Dùng notebook `notebooks/train_on_colab.ipynb`: ô **Data Augmentation** gọi `prepare_augmented_dataset` với đường dẫn `/content/dataset/train/...`, ghi ra `/content/dataset/train_aug`.
- Chi tiết: [docs/COLAB.md](COLAB.md).

### 4.3. Tùy chỉnh pipeline (trong code)

Nếu cần chỉnh cường độ hoặc bỏ/bật từng loại augment:

```python
from src.training.augmentation import (
    AugmentPipeline,
    RandomBrightness,
    RandomHSV,
    # ... import các class cần dùng
)

# Tự tạo pipeline (ví dụ: chỉ ánh sáng + flip)
pipeline = AugmentPipeline(transforms=[
    RandomBrightness(delta=0.2, p=0.6),
    RandomHSV(h=0.02, s=0.3, v=0.3, p=0.5),
    HorizontalFlip(p=0.5),
], seed=42)
```

Pipeline mặc định nằm trong `AugmentPipeline.default_training()` (file `src/training/augmentation.py`), có thể sửa trực tiếp tại đó.

---

## 5. Sử dụng dataset đã augment để train

1. **Tạo hoặc sửa `data.yaml`** (cho Ultralytics):
   - Trỏ `train` tới thư mục ảnh đã augment, ví dụ `train: train_aug/images` (hoặc `train/images` nếu gộp gốc + augment vào một thư mục).
   - `val` vẫn trỏ tới **val gốc** (không augment).

2. **Chạy train YOLO** với `data=` trỏ tới file `data.yaml` đó (local hoặc Colab).

3. Ultralytics vẫn có thể bật thêm augmentation **trong lúc train** (HSV, flip, mosaic, ...). Dataset đã augment offline **bổ sung** đa dạng môi trường (ánh sáng, nhiễu, perspective, cutout), không thay thế hoàn toàn augmentation lúc train.

---

## 6. Lưu ý và khuyến nghị

- **Chỉ augment tập train**, giữ nguyên validation/test để đánh giá công bằng.
- **Số bản augment mỗi ảnh (`--num`)**: 1–3 thường đủ; quá nhiều có thể lặp lại pattern, tăng thời gian train.
- **Seed (`--seed` trong script / `seed` trong pipeline)**: dùng khi cần lặp lại kết quả (debug, so sánh thí nghiệm).
- **Box format**: luôn YOLO chuẩn `[class_id, x_center, y_center, width, height]` normalized; mọi transform hình học đều cập nhật đúng tọa độ.

---

## 7. Tài liệu liên quan

- **API và danh sách transform:** [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md).
- **Train trên Colab (bao gồm bước augmentation):** [COLAB.md](COLAB.md).
