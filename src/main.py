# src/main.py

from src import data_processing
from src import config
from src import feature_engineering
from src import model_training
from src import predict

def main():
    """
    Ana veri bilimi boru hattını (pipeline) çalıştırır.
    """
    # Adım 1: Verileri yükle ve ön işle
    all_data = data_processing.load_all_data()
    all_data = data_processing.preprocess_initial_data(all_data)
    
    if all_data:
        # Adım 2: Eğitim verisi için özellik mühendisliği yap
        # --- DÜZELTİLEN SATIR ---
        train_df_features = feature_engineering.create_features(
            customers=all_data['customers'],
            history=all_data['history'],
            reference_df=all_data['ref_train']
        )
        # -------------------------

        # Adım 3: Veriyi modellemeye hazırla
        X, y = model_training.prepare_data_for_modeling(train_df_features)

        # Adım 4: Veriyi ayır (Optimizasyon ve son skor için)
        X_train, X_val, y_train, y_val = model_training.split_data(X, y)

        # Adım 5: En iyi hiperparametreleri bul
        best_params = model_training.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # --- NİHAİ ADIMLAR ---
        print("\n--- Nihai Model Eğitimi ve Tahmin Süreci ---")
        # Adım 6: Modeli en iyi parametrelerle TÜM eğitim verisiyle eğit
        final_model = model_training.train_final_model(X, y, best_params)

        # Adım 7: Modelin son performansını validasyon setinde gör
        model_training.evaluate_model(final_model, X_val, y_val)

        # Adım 8: Nihai modeli diske kaydet
        model_training.save_model(final_model, config.MODEL_PATH)

        # Adım 9: Test verisi üzerinde tahminleri yap
        predict.make_predictions(final_model, all_data)


if __name__ == '__main__':
    main()