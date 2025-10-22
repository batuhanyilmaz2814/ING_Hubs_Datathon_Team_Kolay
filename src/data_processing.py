# src/data_processing.py

import pandas as pd
from src import config

def load_all_data():
    print("Dosya yolları:")
    print(config.CUSTOMER_HISTORY_PATH)
    print(config.CUSTOMERS_PATH)
    print(config.REFERENCE_DATA_PATH)
    print(config.REFERENCE_DATA_TEST_PATH)
    print(config.SAMPLE_SUBMISSION_PATH)

    """
    Tüm ham veri dosyalarını yükler ve bir sözlük (dictionary) olarak döndürür.
    """
    try:
        data_files = {
            'history': pd.read_csv(config.CUSTOMER_HISTORY_PATH),
            'customers': pd.read_csv(config.CUSTOMERS_PATH),
            'ref_train': pd.read_csv(config.REFERENCE_DATA_PATH),
            'ref_test': pd.read_csv(config.REFERENCE_DATA_TEST_PATH),
            'submission': pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
        }
        print("Tüm ham veri dosyaları başarıyla yüklendi.")
        return data_files
    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı. Lütfen config.py dosyasındaki yolları kontrol edin. Hata: {e}")
        return None

def preprocess_initial_data(data_files):
    """
    Sütun adlarını standartlaştırır ve temel ön işleme adımlarını uygular.
    """
    if data_files:
        # --- Sütun Adlarını Standartlaştırma ---
        # Her DataFrame için 'cust_id' -> 'musteri_id' dönüşümü
        for df_name in ['history', 'customers', 'ref_train', 'ref_test']:
            data_files[df_name].rename(columns={'cust_id': 'musteri_id'}, inplace=True)

        # Tarih ve hedef sütun adlarını standartlaştırma
        data_files['history'].rename(columns={'date': 'ref_dt'}, inplace=True)
        data_files['ref_train'].rename(columns={'ref_date': 'ref_dt', 'churn': config.TARGET_COLUMN}, inplace=True)
        data_files['ref_test'].rename(columns={'ref_date': 'ref_dt'}, inplace=True)
        
        print("Tüm DataFrame'lerde sütun adları standartlaştırıldı.")

        # --- Veri Tipi Dönüşümü ---
        # Artık standart 'ref_dt' ismini güvenle kullanabiliriz.
        data_files['history']['ref_dt'] = pd.to_datetime(data_files['history']['ref_dt'])
        data_files['ref_train']['ref_dt'] = pd.to_datetime(data_files['ref_train']['ref_dt'])
        data_files['ref_test']['ref_dt'] = pd.to_datetime(data_files['ref_test']['ref_dt'])
        print("Tarih sütunları datetime formatına dönüştürüldü.")
        
    return data_files