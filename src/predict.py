# src/predict.py

import pandas as pd
from src import config
from src import feature_engineering
from src import ensemble


def make_predictions(model, all_data):
    pass


def make_predictions_ensemble(lgbm_model, catboost_model, all_data):
    # Bu fonksiyon artık kullanılmıyor, yerine kalibre edilmiş olan kullanılacak
    pass


def make_predictions_calibrated(lgbm_model, catboost_model, calibrator, all_data):
    """
    Test verisi için özellik mühendisliği yapar ve KALİBRE EDİLMİŞ tahminleri oluşturur.
    """
    print("\nTest verisi için KALİBRE EDİLMİŞ Ensemble Tahmin süreci başlıyor...")

    # 1. Test verisi için özellik mühendisliği
    test_df_features = feature_engineering.create_features(
        customers=all_data['customers'],
        history=all_data['history'],
        reference_df=all_data['ref_test']
    )

    # 2. Modelin eğitildiği sütunları belirle
    train_cols = lgbm_model.feature_name_
    X_test = test_df_features.reindex(columns=train_cols, fill_value=0)

    # 3. Ensemble Tahminlerini al
    uncalibrated_preds = ensemble.predict_ensemble(lgbm_model, catboost_model, X_test)

    # 4. Tahminleri Kalibratörden geçir
    predictions = calibrator.predict(uncalibrated_preds)

    # 5. Teslim dosyasını oluştur
    submission_df = pd.DataFrame({
        'cust_id': all_data['ref_test']['musteri_id'],
        'churn': predictions
    })

    # Kaydetme adımları
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"\nTeslim dosyası başarıyla '{config.SUBMISSION_PATH}' olarak oluşturuldu.")
    print("İlk 5 tahmin:")
    print(submission_df.head())