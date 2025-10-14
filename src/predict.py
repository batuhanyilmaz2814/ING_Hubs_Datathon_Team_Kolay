# src/predict.py

import pandas as pd
from src import config  # <-- config'i import ediyoruz
from src import feature_engineering

def make_predictions(model, all_data):
    """
    Test verisi için özellik mühendisliği yapar ve tahminleri oluşturur.
    """
    print("\nTest verisi için tahmin süreci başlıyor...")

    # 1. Test verisi için özellik mühendisliği
    test_df_features = feature_engineering.create_features(
        customers=all_data['customers'],
        history=all_data['history'],
        reference_df=all_data['ref_test']
    )

    # --- DÜZELTME BURADA ---
    # Model tipine göre doğru özellik adı listesini alıyoruz.
    if config.MODEL_TYPE == 'lightgbm':
        train_cols = model.feature_name_
    elif config.MODEL_TYPE == 'xgboost':
        train_cols = model.feature_names_in_
    else:
        raise ValueError(f"Bilinmeyen model tipi: {config.MODEL_TYPE}")
    # ---------------------

    # Test setinin sütunlarını, eğitim setinin sütunlarıyla tam olarak aynı hale getir.
    X_test = test_df_features.reindex(columns=train_cols, fill_value=0)
    
    # 3. Tahminleri yap
    predictions = model.predict_proba(X_test)[:, 1]

    # 4. Teslim dosyasını oluştur
    submission_df = pd.DataFrame({
        'cust_id': all_data['ref_test']['musteri_id'],
        'churn': predictions
    })
    
    # 5. Dosyayı kaydet
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"\nTeslim dosyası başarıyla '{config.SUBMISSION_PATH}' olarak oluşturuldu.")
    print("İlk 5 tahmin:")
    print(submission_df.head())