# Retail Demand Forecasting

A comprehensive machine learning system designed to forecast product demand for retail operations. By accurately predicting future sales volumes based on historical data and exogenous variables (such as weather, promotions, and holidays), this system helps businesses optimize their inventory, reduce stockouts, and minimize waste.

---
## Report
- Visit docs/report.md to view the report written by me

## 🚀 Key Features
- **Automated Data Pipeline:** Fetches the `Dingdong-Inc/FreshRetailNet-50K` dataset directly from Hugging Face Hub, cleans it, and handles missing weather values using linear interpolation.
- **Advanced Time-Series Modeling:** Uses `MLForecast` to automatically generate highly predictive lag features and rolling window statistics without data leakage.
- **Multiple Algorithms:** Trains and evaluates Ridge Regression, XGBoost, and LightGBM concurrently.
- **Robust Evaluation:** Employs time-series cross-validation to rigorously validate model performance.
- **Production-Ready Serving:** Deploys the best model behind a scalable FastAPI layer orchestrated by Ray Serve.

---

## 🛠 Tech Stack
- **Data Manipulation:** `pandas`, `numpy`, `datasets`
- **Machine Learning:** `mlforecast`, `xgboost`, `lightgbm`, `scikit-learn`
- **Serving & Deployment:** `FastAPI`, `Ray Serve`, `uvicorn`

---

## 📁 Project Structure

```text
├── data/                  # Data directories
│   ├── raw/               # Cleaned data (Output of data_preparation)
│   └── processed/         # Feature engineered data (Legacy)
├── models/                # Trained models and evaluation metrics
│   └── mlforecast/        # Saved MLForecast objects and results
├── src/                   # Source code
│   ├── data_preparation.py     # Download, clean and interpolate data
│   ├── mlforecast_train.py     # Primary training script using MLForecast
│   ├── serve.py                # FastAPI endpoints
│   └── ray_serve.py            # Ray Serve deployment wrapper
└── README.md
```


---

## 💻 How to Run

### 1. Environment Setup
Make sure you have Python installed, then create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies (assuming requirements.txt exists)
pip install -r requirements.txt
```

### 2. Data Preparation
Run the data preparation script to download the dataset from Hugging Face and clean it. You can optionally pass `--city_id` to filter data for a specific city to save RAM.
```bash
python src/data_preparation.py --city_id 0
```
*Output: `train_clean.parquet` and `eval_clean.parquet` in `data/raw/`.*

### 3. Model Training
Train the models using the `MLForecast` framework. This script builds lag features, performs time-series cross-validation, trains multiple models, and evaluates them.
```bash
python src/mlforecast_train.py --city_id 0 --horizon 7
```
If you want to skip cross-validation for a quicker run, add `--no_cv`:
```bash
python src/mlforecast_train.py --city_id 0 --horizon 7 --no_cv
```

### 4. Serving (API Deployment)
Start the Ray Serve application to host the trained models via FastAPI.
```bash
python src/ray_serve.py
```
Once the server is running, you can access the interactive API documentation (Swagger UI) at:
[http://localhost:8000/docs](http://localhost:8000/docs)

===========================================================================
===========================================================================

# Retail Demand Forecasting (Dự báo Nhu cầu Bán lẻ)

Đây là hệ thống Machine Learning hoàn chỉnh được thiết kế để dự báo nhu cầu sản phẩm cho các doanh nghiệp bán lẻ. Bằng cách dự báo chính xác lượng bán dựa trên dữ liệu lịch sử và các biến số ngoại lai (như thời tiết, khuyến mãi, ngày lễ), hệ thống giúp doanh nghiệp tối ưu hóa hàng tồn kho, giảm thiểu tình trạng hết hàng và hạn chế lãng phí.

---

## 🚀 Tính năng chính
- **Pipeline Dữ liệu Tự động:** Tải trực tiếp bộ dữ liệu `Dingdong-Inc/FreshRetailNet-50K` từ Hugging Face, làm sạch và xử lý các giá trị thời tiết bị thiếu bằng phương pháp nội suy tuyến tính (linear interpolation).
- **Mô hình Chuỗi thời gian Tiên tiến:** Sử dụng framework `MLForecast` để tự động sinh ra các đặc trưng trễ (lag features) và trung bình trượt (rolling window) tối ưu mà không lo rò rỉ dữ liệu.
- **Đa Thuật toán:** Huấn luyện và đánh giá song song Ridge Regression, XGBoost, và LightGBM.
- **Đánh giá Chặt chẽ:** Sử dụng phương pháp Time-Series Cross Validation để kiểm định độ ổn định của mô hình.
- **Sẵn sàng cho Production:** Triển khai API cho mô hình tốt nhất thông qua FastAPI, được quản lý, mở rộng (scale) linh hoạt bằng Ray Serve.

---

## 🛠 Công nghệ sử dụng
- **Xử lý dữ liệu:** `pandas`, `numpy`, `datasets`
- **Machine Learning:** `mlforecast`, `xgboost`, `lightgbm`, `scikit-learn`
- **Triển khai (Serving):** `FastAPI`, `Ray Serve`, `uvicorn`

---

## 📁 Cấu trúc thư mục

```text
├── data/                  # Nơi lưu trữ dữ liệu
│   ├── raw/               # Dữ liệu sạch (Đầu ra của data_preparation)
│   └── processed/         # Dữ liệu sau feature engineering (Legacy)
├── models/                # Nơi lưu mô hình đã train và kết quả đánh giá
│   └── mlforecast/        # Đối tượng MLForecast và model_info.json
├── src/                   # Source code
│   ├── data_preparation.py     # Tải, làm sạch và nội suy dữ liệu thời tiết
│   ├── mlforecast_train.py     # Script train tự động hóa bằng MLForecast
│   ├── serve.py                # Định nghĩa các endpoint FastAPI
│   └── ray_serve.py            # Chạy server FastAPI thông qua Ray Serve
└── README.md
```

*(Lưu ý: `src/feature_engineering.py` và `src/train.py` là một phần của luồng pipeline cũ và đã được thay thế bởi `mlforecast_train.py`.)*

---

## 💻 Hướng dẫn chạy

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python. Tạo và kích hoạt môi trường ảo (virtual environment):
```bash
python -m venv .venv
# Trên Windows:
.\.venv\Scripts\activate
# Trên macOS/Linux:
source .venv/bin/activate

# Cài đặt các thư viện phụ thuộc (giả sử bạn có file requirements.txt)
pip install -r requirements.txt
```

### 2. Tiền xử lý dữ liệu (Data Preparation)
Chạy script này để tải dữ liệu thô từ Hugging Face và làm sạch. Bạn có thể truyền cờ `--city_id` để chỉ xử lý dữ liệu của một thành phố cụ thể (giúp tiết kiệm RAM máy tính).
```bash
python src/data_preparation.py --city_id 0
```
*Kết quả sẽ sinh ra các file `train_clean.parquet` và `eval_clean.parquet` trong thư mục `data/raw/`.*

### 3. Huấn luyện mô hình (Training)
Sử dụng script MLForecast để tự động hóa hoàn toàn quá trình tạo đặc trưng, chạy Cross Validation, huấn luyện và đánh giá mô hình.
```bash
python src/mlforecast_train.py --city_id 0 --horizon 7
```
Nếu bạn muốn quá trình chạy nhanh hơn (bỏ qua bước đánh giá chéo Cross-Validation), hãy thêm cờ `--no_cv`:
```bash
python src/mlforecast_train.py --city_id 0 --horizon 7 --no_cv
```

### 4. Triển khai API (Serving)
Chạy ứng dụng Ray Serve để khởi động các endpoint FastAPI cung cấp dịch vụ dự báo.
```bash
python src/ray_serve.py
```
Sau khi server báo đã khởi động thành công, bạn có thể truy cập trang tài liệu API tương tác (Swagger UI) tại đường dẫn:
[http://localhost:8000/docs](http://localhost:8000/docs)
