# src/config.py

# --- Dosya Yolları ---
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

CUSTOMER_HISTORY_PATH = f"{RAW_DATA_DIR}/customer_history.csv"
CUSTOMERS_PATH = f"{RAW_DATA_DIR}/customers.csv"
REFERENCE_DATA_PATH = f"{RAW_DATA_DIR}/referance_data.csv"
REFERENCE_DATA_TEST_PATH = f"{RAW_DATA_DIR}/referance_data_test.csv"
SAMPLE_SUBMISSION_PATH = f"{RAW_DATA_DIR}/sample_submission.csv"

# --- Çıktı Dosya Yolları ---
MODEL_PATH = f"{MODELS_DIR}/model.txt"
SUBMISSION_PATH = "submission.csv"

# --- Model Ayarları ---
# Hangi modeli kullanmak istediğinizi buradan seçin: 'lightgbm' veya 'xgboost'
MODEL_TYPE = 'lightgbm' 

TARGET_COLUMN = 'target'
RANDOM_STATE = 42