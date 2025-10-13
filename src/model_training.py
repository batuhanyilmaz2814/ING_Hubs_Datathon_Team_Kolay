# src/model_training.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src import config
import joblib

def prepare_data_for_modeling(df):
    """
    Özellik matrisini modellemeye hazır hale getirir (X ve y olarak ayırır).
    """
    # Modelin kullanmayacağı, tanımlayıcı sütunları kaldıralım.
    # 'province' gibi one-hot encode edilmemiş kategorik sütunlar varsa onlar da kaldırılmalı.
    features_to_drop = [config.TARGET_COLUMN, 'musteri_id', 'ref_dt', 'province']
    
    # province sütunu yoksa hata vermemesi için kontrol edelim
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]

    X = df.drop(columns=existing_cols_to_drop)
    y = df[config.TARGET_COLUMN]
    
    return X, y

def split_data(X, y):
    """
    Veriyi eğitim ve validasyon setlerine ayırır.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y  # Dengesiz veri setleri için target dağılımını korur
    )
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Validasyon seti boyutu: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    """
    LightGBM modelini eğitir.
    """
    print("\nModel eğitimi başlıyor...")
    model = lgb.LGBMClassifier(random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)
    print("Model eğitimi tamamlandı.")
    return model

def evaluate_model(model, X_val, y_val):
    """
    Modeli yarışmanın özel metriklerine göre değerlendirir.
    """
    print("\nModel değerlendiriliyor...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Baseline değerleri
    baseline_gini = 0.38515
    baseline_recall_at_10 = 0.18469
    baseline_lift_at_10 = 1.84715

    # 1. Gini Hesaplama
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    gini = 2 * roc_auc - 1

    # 2. Recall@10% ve Lift@10% Hesaplama
    df_eval = pd.DataFrame({'true': y_val, 'proba': y_pred_proba})
    df_eval = df_eval.sort_values('proba', ascending=False)
    
    top_10_percent_count = int(len(df_eval) * 0.1)
    top_10_df = df_eval.head(top_10_percent_count)
    
    total_positives = df_eval['true'].sum()
    top_10_positives = top_10_df['true'].sum()
    
    recall_at_10 = top_10_positives / total_positives if total_positives > 0 else 0
    lift_at_10 = recall_at_10 / 0.1 if recall_at_10 > 0 else 0
    
    # 3. Final Yarışma Skoru
    score = (0.4 * (gini / baseline_gini) +
             0.3 * (recall_at_10 / baseline_recall_at_10) +
             0.3 * (lift_at_10 / baseline_lift_at_10))
    
    print("\n--- DEĞERLENDİRME SONUÇLARI ---")
    print(f"ROC AUC: {roc_auc:.5f}")
    print(f"Gini: {gini:.5f} (Baseline: {baseline_gini:.5f})")
    print(f"Recall@10%: {recall_at_10:.5f} (Baseline: {baseline_recall_at_10:.5f})")
    print(f"Lift@10%: {lift_at_10:.5f} (Baseline: {baseline_lift_at_10:.5f})")
    print("-" * 30)
    print(f"Final Yarışma Skoru: {score:.5f}")
    print("---------------------------------")

    
    return score

def save_model(model, file_path):
    """
    Eğitilmiş modeli diske kaydeder.
    """
    print(f"Model {file_path} adresine kaydediliyor...")
    joblib.dump(model, file_path)
    print("Model başarıyla kaydedildi.")