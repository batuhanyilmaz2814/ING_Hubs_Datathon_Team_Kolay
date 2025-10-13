# src/main.py

from src import data_processing
from src import config
from src import feature_engineering
from src import model_training
from src import predict # Yeni modülümüzü import ediyoruz

def main():
    """
    Ana veri bilimi boru hattını (pipeline) çalıştırır.
    """
    # Adım 1: Verileri yükle ve ön işle
    all_data = data_processing.load_all_data()
    all_data = data_processing.preprocess_initial_data(all_data)
    
    if all_data:
        # Adım 2: Eğitim verisi için özellik mühendisliği yap
        train_df_features = feature_engineering.create_features(
            customers=all_data['customers'],
            history=all_data['history'],
            reference_df=all_data['ref_train']
        )

        # Adım 3: Veriyi modellemeye hazırla
        X, y = model_training.prepare_data_for_modeling(train_df_features)

        # Adım 4: Veriyi eğitim ve validasyon setlerine ayır (skoru görmek için)
        X_train, X_val, y_train, y_val = model_training.split_data(X, y)

        # Adım 5: Modeli eğit ve değerlendir
        model = model_training.train_model(X_train, y_train)
        model_training.evaluate_model(model, X_val, y_val)
        
        # --- NİHAİ ADIMLAR ---
        print("\n--- Nihai Model Eğitimi ve Tahmin Süreci ---")
        # Adım 6: Modeli tüm eğitim verisiyle yeniden eğit
        final_model = model_training.train_model(X, y)

        # Adım 7: Nihai modeli diske kaydet
        model_training.save_model(final_model, config.MODEL_PATH)

        # Adım 8: Test verisi üzerinde tahminleri yap ve teslim dosyasını oluştur
        predict.make_predictions(final_model, all_data)


if __name__ == '__main__':
    main()