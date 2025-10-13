# src/config.py

# --- Dosya Yolları ---
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Ham veri dosyalarının tam yolları
CUSTOMER_HISTORY_PATH = f"{RAW_DATA_DIR}/customer_history.csv"
CUSTOMERS_PATH = f"{RAW_DATA_DIR}/customers.csv"
REFERENCE_DATA_PATH = f"{RAW_DATA_DIR}/referance_data.csv"
REFERENCE_DATA_TEST_PATH = f"{RAW_DATA_DIR}/referance_data_test.csv"
SAMPLE_SUBMISSION_PATH = f"{RAW_DATA_DIR}/sample_submission.csv"

# --- Model Ayarları ---
TARGET_COLUMN = 'target'
# Diğer model parametreleri buraya eklenecek...