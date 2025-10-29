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
    print("Adım 1: Veriler yükleniyor ve ön işleniyor...")
    all_data = data_processing.load_all_data()
    all_data = data_processing.preprocess_initial_data(all_data)
    
    if all_data:
        print("\nAdım 2: Yüksek etkili özellikler oluşturuluyor...")
        train_df_features = feature_engineering.create_features(
            customers=all_data['customers'],
            history=all_data['history'],
            reference_df=all_data['ref_train']
        )

        print("\nAdım 3: Veri, zaman odaklı olarak eğitim ve validasyon setlerine ayrılıyor...")
        X_train, X_val, y_train, y_val = model_training.split_data_by_time(train_df_features)

        print("\nAdım 4: En iyi model parametreleri Optuna ile bulunuyor...")
        best_params = model_training.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        print("\n--- Nihai Model Eğitimi ve Tahmin Süreci Başlıyor ---")

        X_final, y_final = model_training.prepare_data_for_modeling(train_df_features)
        
        final_model = model_training.train_final_model(X_final, y_final, best_params)

        model_training.evaluate_model(final_model, X_val, y_val)

        model_training.save_model(final_model, config.MODEL_PATH)

        predict.make_predictions(final_model, all_data)


if __name__ == '__main__':
    main()