# src/main.py

from src import data_processing
from src import config

def initial_data_analysis(data_files):
    """
    Yüklenen verilerin temel analizlerini yapar ve ekrana yazdırır.
    """
    print("\n--- Müşteri Demografik Bilgileri Detayları ---")
    print(data_files['customers'].info())
    print("\nEksik Veri Sayısı:")
    print(data_files['customers'].isnull().sum())
    
    print("\n--- Müşteri İşlem Geçmişi Detayları ---")
    print(data_files['history'].info())
    print("\nEksik Veri Sayısı:")
    print(data_files['history'].isnull().sum())

    print("\n--- Eğitim Referans Verisi (Churn) Dağılımı ---")
    churn_dist = data_files['ref_train'][config.TARGET_COLUMN].value_counts(normalize=True) * 100
    print(churn_dist)

if __name__ == '__main__':
    # Adım 1: Verileri yükle
    all_data = data_processing.load_all_data()
    
    # Adım 2: Temel ön işleme
    all_data = data_processing.preprocess_initial_data(all_data)
    
    # Adım 3: İlk veri analizini yap
    if all_data:
        initial_data_analysis(all_data)