# src/model_training.py

import joblib
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from src import config
import optuna
from src.metrics import ing_hubs_datathon_metric

def prepare_data_for_modeling(df):
    features_to_drop = [config.TARGET_COLUMN, 'musteri_id', 'ref_dt', 'province']
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    y = df[config.TARGET_COLUMN]
    return X, y

def split_data_by_time(df_features):
    print("Veri, zaman odaklı olarak ayrılıyor...")
    split_date = df_features['ref_dt'].max() - pd.DateOffset(months=3)
    train_df = df_features[df_features['ref_dt'] < split_date].copy()
    val_df = df_features[df_features['ref_dt'] >= split_date].copy()
    X_train, y_train = prepare_data_for_modeling(train_df)
    X_val, y_val = prepare_data_for_modeling(val_df)
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Validasyon seti boyutu: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    print(f"\n{config.MODEL_TYPE.upper()} için HIZLI ve GÜVENLİ hiperparametre optimizasyonu başlıyor...")
    def objective(trial):
        # Aşırı öğrenmeyi önlemek için daha basit ve kısıtlı bir arama uzayı
        params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'n_estimators': 1000, # Sabit ve yüksek tutuyoruz, early stopping halledecek
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 10, 40),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        }
        model = lgb.LGBMClassifier(**params, random_state=config.RANDOM_STATE)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        preds = model.predict_proba(X_val)[:, 1]
        score = ing_hubs_datathon_metric(y_val, preds)
        return score
        
    study = optuna.create_study(direction='maximize')
    # Daha hızlı sonuç için deneme sayısını 15'e düşürelim
    study.optimize(objective, n_trials=15) 
    print(f"En iyi deneme skoru: {study.best_value}")
    print(f"En iyi parametreler: {study.best_params}")
    return study.best_params

# Diğer fonksiyonlar (train_final_model, evaluate_model, save_model) aynı kalabilir.
# Sadece config'den MODEL_TYPE='lightgbm' seçtiğinizden emin olun.
# ... (geri kalan fonksiyonları silmeyin, aynı kalacaklar) ...
def train_final_model(X, y, best_params):
    print(f"\nNihai model ({config.MODEL_TYPE.upper()}) en iyi parametrelerle eğitiliyor...")
    model = lgb.LGBMClassifier(**best_params, n_estimators=1000, random_state=config.RANDOM_STATE)
    model.fit(X, y)
    print("Nihai model eğitimi tamamlandı.")
    return model

def evaluate_model(model, X_val, y_val):
    print("\nModel resmi metrik fonksiyonu ile değerlendiriliyor...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    final_score = ing_hubs_datathon_metric(y_val, y_pred_proba)
    print(f"\n--- DEĞERLENDİRME SONUÇLARI (Resmi Metrik) ---")
    print(f"Final Resmi Yarışma Skoru: {final_score:.5f}")
    print("---------------------------------------------")
    return final_score

def save_model(model, file_path):
    model_path = file_path.replace('.txt', f'_{config.MODEL_TYPE}.txt')
    joblib.dump(model, model_path)
    print(f"Model başarıyla {model_path} olarak kaydedildi.")