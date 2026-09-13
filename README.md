# Support Vector Machine - Demo dự đoán giá nhà

## 1. Giới thiệu

Đây là demo môn Học máy cơ bản sử dụng thuật toán Support Vector Machine (SVM) trên dữ liệu giá nhà.

Demo thực hiện hai bài toán:

* Phân loại nhà theo nhóm giá bằng Support Vector Classification.
* Dự đoán giá nhà bằng Support Vector Regression.

Mô hình được triển khai thành REST API bằng FastAPI và đóng gói bằng Docker.

## 2. Dataset

Dataset được sử dụng là dữ liệu giá nhà được lấy từ Kaggle.

Các thuộc tính đầu vào:

* `bedrooms`: Số phòng ngủ.
* `bathrooms`: Số phòng tắm.
* `sqft_living`: Diện tích khu vực sinh hoạt.
* `grade`: Mức đánh giá chất lượng căn nhà.
* `yr_built`: Năm xây dựng.

Thuộc tính mục tiêu:

* `price`: Giá nhà.

## 3. Mô hình

### Support Vector Classification

Sử dụng Support Vector Classification với:

* Kernel: RBF
* C = 1.0
* Gamma = scale

Giá nhà được chia thành 3 nhóm:

* Thấp
* Trung bình
* Cao

### Support Vector Regression

Sử dụng Support Vector Regression với:

* Kernel: RBF
* C = 100
* Gamma = scale
* Epsilon = 0.1

Trước khi đưa dữ liệu vào mô hình, dữ liệu được chuẩn hóa bằng StandardScaler.

## 4. API

API được xây dựng bằng FastAPI.

### Kiểm tra API

```text
GET /
```

### Phân loại nhóm giá

```text
POST /predict-classification
```

Dữ liệu đầu vào:

```json
[
  3,
  2.0,
  1800,
  7,
  1995
]
```

Kết quả ví dụ:

```json
{
  "prediction": [
    "Thấp"
  ]
}
```

### Dự đoán giá nhà

```text
POST /predict-regression
```

Dữ liệu đầu vào:

```json
[
  3,
  2.0,
  1800,
  7,
  1995
]
```

Kết quả trả về là giá nhà dự đoán.

## 5. Cấu trúc project

```text
Support-Vector/
│
├── app.py
├── Dockerfile
├── requirements.txt
│
├── svc_model.pkl
├── svr_model.pkl
├── scaler_svc.pkl
└── scaler_svr.pkl
```

## 6. Cài đặt và chạy

Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

Chạy API:

```bash
uvicorn app:app --reload
```

API chạy tại:

```text
http://127.0.0.1:8000
```

Swagger API:

```text
http://127.0.0.1:8000/docs
```

## 7. Chạy bằng Docker

Build Docker image:

```bash
docker build -t support-vector-api .
```

Chạy container:

```bash
docker run -d -p 8000:8000 --name support-vector-api support-vector-api
```

Sau đó truy cập:

```text
http://127.0.0.1:8000/docs
```

## 8. Công nghệ sử dụng

* Python
* Scikit-learn
* FastAPI
* Uvicorn
* NumPy
* Joblib
* Docker
* Kaggle
