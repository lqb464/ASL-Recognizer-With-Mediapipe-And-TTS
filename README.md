# README (Tiếng Việt)

English version: README-en.md

## Giới thiệu

ASL-TALK là một dự án Machine Learning nhằm nhận diện ký hiệu ngôn ngữ ký hiệu (American Sign Language - ASL) từ dữ liệu video webcam theo thời gian thực. Hệ thống sử dụng hand landmarks để biểu diễn chuyển động tay, sau đó huấn luyện một mô hình sequence model để dự đoán ký hiệu tương ứng.

Mục tiêu của dự án là xây dựng một pipeline ML hoàn chỉnh từ thu thập dữ liệu, xử lý dữ liệu, huấn luyện mô hình cho tới suy luận thời gian thực.

Dự án này được xây dựng như một portfolio project nhằm thể hiện khả năng thiết kế pipeline machine learning, tổ chức code và xây dựng hệ thống inference đơn giản.

---

## Tính năng chính

Thu thập dữ liệu ký hiệu từ webcam
Nhập dữ liệu video từ nguồn bên ngoài
Trích xuất hand landmarks từ video
Chuẩn hóa và xây dựng dataset huấn luyện
Huấn luyện mô hình sequence để nhận diện ký hiệu
Chạy inference thời gian thực từ webcam
Tổ chức project theo cấu trúc ML pipeline chuẩn

---

## Cấu trúc project

```
ASL-TALK
│
├─ configs
│
├─ data
│  ├─ raw
│  ├─ interim
│  ├─ processed
│  └─ external
│
├─ models
│  ├─ checkpoints
│  └─ trained
│
├─ src
│  ├─ data
│  ├─ models
│  ├─ pipelines
│  ├─ utils
│  └─ __init__.py
│
├─ tests
│
├─ pyproject.toml
├─ README.md
├─ README-en.md   # đây là version README.md tiếng Anh
├─ LICENSE
└─ .gitignore
```

---

## Pipeline Machine Learning

Quy trình xử lý dữ liệu và huấn luyện mô hình bao gồm các bước sau:

1 Thu thập dữ liệu từ webcam
2 Lưu dữ liệu thô vào thư mục raw
3 Chuyển dữ liệu raw thành dữ liệu interim
4 Chuẩn hóa dữ liệu thành dataset processed
5 Huấn luyện mô hình sequence
6 Lưu checkpoint mô hình
7 Chạy inference từ webcam

Luồng pipeline:

```
data collection
      ↓
raw dataset
      ↓
dataset preprocessing
      ↓
processed dataset
      ↓
model training
      ↓
model checkpoint
      ↓
real-time inference
```

---

## Cài đặt môi trường

Yêu cầu Python 3.10 trở lên.

Cài đặt dependencies:

```
pip install -e .
```

---

## Xây dựng dataset

Chạy pipeline xây dựng dataset:

```
python -m pipelines.run_dataset
```

Pipeline này sẽ thực hiện:

- raw_to_interim.py
- interim_to_processed

Sau khi chạy xong, dataset huấn luyện sẽ được tạo trong thư mục:

```
data/processed
```

---

## Huấn luyện mô hình

Chạy pipeline training:

```
python -m pipelines.run_training
```

Sau khi huấn luyện hoàn tất, checkpoint mô hình sẽ được lưu tại:

```
models/checkpoints
```

---

## Chạy inference

Chạy inference từ webcam:

```
python -m src.tests.test_infer_webcam
```

Hệ thống sẽ:

1 Mở webcam
2 Phát hiện bàn tay
3 Trích xuất landmarks
4 Dự đoán ký hiệu bằng mô hình đã huấn luyện

---

## Cấu hình

Các tham số của hệ thống được quản lý thông qua các file YAML trong thư mục:

```
configs
```

Ví dụ:

```
data.yaml
model.yaml
train.yaml
utils.yaml
```

Các file này cho phép điều chỉnh cấu hình mà không cần sửa code.

---

## Mục tiêu học tập

Dự án này nhằm thể hiện khả năng:

thiết kế ML pipeline
xây dựng dataset pipeline
huấn luyện mô hình sequence
tổ chức code theo cấu trúc ML project chuẩn
xây dựng inference pipeline đơn giản

---

## License

Dự án được phát hành dưới giấy phép MIT.
