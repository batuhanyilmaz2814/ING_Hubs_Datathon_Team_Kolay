# src/model_training.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src import config
import joblib
import optuna

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


def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Optuna kullanarak en iyi LightGBM hiperparametrelerini bulur.
    """
    print("\nHiperparametre optimizasyonu başlıyor...")

    def objective(trial):
        # Ayarlanacak parametre aralıklarını tanımla
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        model = lgb.LGBMClassifier(**params, random_state=config.RANDOM_STATE)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # 100 deneme boyunca skor artmazsa dur
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # 50 farklı parametre kombinasyonu deneyecek

    print("En iyi deneme:")
    print(f"  Değer (AUC): {study.best_value}")
    print("  Parametreler: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    return study.best_params

def train_final_model(X, y, best_params):
    """
    Bulunan en iyi parametrelerle nihai modeli eğitir.
    """
    print("\nNihai model en iyi parametrelerle eğitiliyor...")
    final_model = lgb.LGBMClassifier(**best_params, random_state=config.RANDOM_STATE)
    final_model.fit(X, y)
    print("Nihai model eğitimi tamamlandı.")
    return final_model

# src/model_training.py -> Bu fonksiyonu güncelliyoruz

# Yeni metrik modülümüzü ve içindeki fonksiyonları import edelim
from src.metrics import ing_hubs_datathon_metric, recall_at_k, lift_at_k, convert_auc_to_gini

def evaluate_model(model, X_val, y_val):
    """
    Modeli YARIŞMANIN RESMİ METRİĞİNE göre değerlendirir.
    """
    print("\nModel resmi metrik fonksiyonu ile değerlendiriliyor...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Resmi fonksiyonu kullanarak nihai skoru hesapla
    final_score = ing_hubs_datathon_metric(y_val, y_pred_proba)

    # Ekrana detaylı bilgi basmak için bireysel metrikleri de hesaplayalım
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    gini = convert_auc_to_gini(roc_auc)
    recall = recall_at_k(y_val, y_pred_proba)
    lift = lift_at_k(y_val, y_pred_proba)

    print("\n--- DEĞERLENDİRME SONUÇLARI (Resmi Metrik) ---")
    print(f"ROC AUC: {roc_auc:.5f}")
    print(f"Gini: {gini:.5f}")
    print(f"Recall@10%: {recall:.5f}")
    print(f"Lift@10%: {lift:.5f}")
    print("-" * 30)
    print(f"Final Resmi Yarışma Skoru: {final_score:.5f}")
    print("---------------------------------------------")
    
    # Optuna gibi yerlerde kullanılmak üzere nihai skoru döndür
    return final_score

def save_model(model, file_path):
    """
    Eğitilmiş modeli diske kaydeder.
    """
    print(f"Model {file_path} adresine kaydediliyor...")
    joblib.dump(model, file_path)
    print("Model başarıyla kaydedildi.")