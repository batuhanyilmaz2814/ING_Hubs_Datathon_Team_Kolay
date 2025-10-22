# src/main.py

from src import data_processing
from src import config
from src import feature_engineering
from src import model_training
from src import predict
from src import ensemble
from src import reporting
from src.metrics import ing_hubs_datathon_metric


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

        # Adım 3: Veriyi hazırla
        X, y, categorical_features = model_training.prepare_data_for_modeling(train_df_features)

        # Adım 4 & 5: Parametre Optimizasyonu ve Veri Ayırma
        # OPTİMİZASYON İÇİN TÜM X VE Y GÖNDERİLİR
        best_params_lgbm = model_training.optimize_hyperparameters(X, y, categorical_features)

        # MODEL EĞİTİMİ İÇİN TEKRAR AYIRMA
        X_train, X_val, y_train, y_val, categorical_features = model_training.split_data(X, y, categorical_features)

        # --- FİNAL ENSEMBLE (CATBOOST + LIGHTGBM) EĞİTİMİ VE TAHMİN SÜRECİ ---
        print("\n--- FİNAL ENSEMBLE (CATBOOST + LIGHTGBM) EĞİTİMİ VE TAHMİN SÜRECİ ---")

        # 6. TEMEL MODELLERİN EĞİTİMİ (X_train üzerinde)
        model_lgbm_eval, best_n_estimators = model_training.train_lgbm_with_early_stopping(
            X_train, y_train, X_val, y_val, best_params_lgbm, categorical_features
        )
        catboost_model_for_evaluation = ensemble.train_catboost(X_train, y_train, categorical_features)

        # 7a. Ensemble Tahminini al
        lgbm_preds_val = model_lgbm_eval.predict_proba(X_val)[:, 1]
        catboost_preds_val = catboost_model_for_evaluation.predict_proba(X_val)[:, 1]

        # ENSEMBLE AĞIRLIĞINI OPTİMİZE ET
        optimized_weight_cb, uncalibrated_preds = ensemble.optimize_ensemble_weight(
            lgbm_preds_val, catboost_preds_val, y_val
        )
        print(f"-> Optimizasyon Sonucu En İyi CatBoost Ağırlığı: {optimized_weight_cb:.2f}")

        # 7b. Kalibrasyon Modelini Eğit
        calibrator = ensemble.calibrate_predictions(uncalibrated_preds, y_val)

        # 7c. Tahminleri Kalibre Et ve Skoru Hesapla
        ensemble_preds = calibrator.predict(uncalibrated_preds)

        final_ensemble_score = ing_hubs_datathon_metric(y_val, ensemble_preds)

        # --- KAPSAMLI GERİ BİLDİRİM ÇAĞRISI ---
        base_models_eval = {
            'lgbm': model_lgbm_eval,
            'catboost': catboost_model_for_evaluation
        }
        reporting.generate_performance_report(base_models_eval, X_val, y_val, final_ensemble_score)

        # 8. MODELLERİ TÜM X ve Y ÜZERİNDE SON KEZ EĞİT
        print("\nNihai modeller, Test Tahmini için TÜM VERİ ÜZERİNDE son kez eğitiliyor...")

        final_lgbm_full = model_training.train_final_model_full_data(
            X, y, best_params_lgbm, categorical_features, best_n_estimators
        )
        final_catboost_full = ensemble.train_catboost(X, y, categorical_features)

        # Final Kalibrasyon Modeli Eğitimi
        full_ensemble_preds = ensemble.predict_ensemble(final_lgbm_full, final_catboost_full, X, optimized_weight_cb)
        final_calibrator = ensemble.calibrate_predictions(full_ensemble_preds, y)

        model_training.save_model(final_lgbm_full, config.MODEL_PATH)

        # 9. Test verisi üzerinde Ensemble tahminlerini yap
        predict.make_predictions_calibrated(final_lgbm_full, final_catboost_full, final_calibrator, all_data,
                                            optimized_weight_cb)


if __name__ == '__main__':
    main()