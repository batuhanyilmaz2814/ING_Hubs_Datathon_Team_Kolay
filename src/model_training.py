# src/model_training.py

import joblib
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
# Artık xgboost.callback importuna gerek yok
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src import config
import optuna
from src.metrics import ing_hubs_datathon_metric, recall_at_k, lift_at_k, convert_auc_to_gini

# ... (prepare_data_for_modeling ve split_data fonksiyonları aynı kalacak) ...
def prepare_data_for_modeling(df):
    features_to_drop = [config.TARGET_COLUMN, 'musteri_id', 'ref_dt', 'province']
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    y = df[config.TARGET_COLUMN]
    return X, y

def split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Validasyon seti boyutu: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Optuna kullanarak config'de seçilen model için en iyi hiperparametreleri bulur.
    """
    print(f"\n{config.MODEL_TYPE.upper()} için hiperparametre optimizasyonu başlıyor...")

    def objective(trial):
        if config.MODEL_TYPE == 'lightgbm':
            # ... (LightGBM kısmı aynı kalacak) ...
            params = {
                'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 15, 80),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            }
            model = lgb.LGBMClassifier(**params, random_state=config.RANDOM_STATE)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        
        elif config.MODEL_TYPE == 'xgboost':
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0,
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
            }
            
            # --- KESİN ÇÖZÜM BURADA ---
            # Early stopping parametresini doğrudan modelin kendisine veriyoruz.
            model = xgb.XGBClassifier(**params, 
                                      early_stopping_rounds=100, # <-- DOĞRU YÖNTEM
                                      random_state=config.RANDOM_STATE)
            
            # fit() metodundan ilgili parametreleri kaldırıyoruz.
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            # --------------------------

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("En iyi deneme:")
    print(f"  Değer (AUC): {study.best_value}")
    print("  Parametreler: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    return study.best_params

def train_final_model(X, y, best_params):
    print(f"\nNihai model ({config.MODEL_TYPE.upper()}) en iyi parametrelerle eğitiliyor...")
    
    if config.MODEL_TYPE == 'lightgbm':
        model = lgb.LGBMClassifier(**best_params, random_state=config.RANDOM_STATE)
    elif config.MODEL_TYPE == 'xgboost':
        if 'use_label_encoder' in best_params:
            del best_params['use_label_encoder']
        model = xgb.XGBClassifier(**best_params, random_state=config.RANDOM_STATE)
        
    model.fit(X, y)
    print("Nihai model eğitimi tamamlandı.")
    return model

def evaluate_model(model, X_val, y_val):
    """
    Modeli YARIŞMANIN RESMİ METRİĞİNE göre değerlendirir.
    """
    print("\nModel resmi metrik fonksiyonu ile değerlendiriliyor...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Resmi fonksiyonu kullanarak nihai skoru hesapla
    final_score = ing_hubs_datathon_metric(y_val, y_pred_proba)

    # Ekrana detaylı bilgi basmak için bireysel metrikleri de hesaplayalım
    # Bu metrikler artık src/metrics.py içerisinden çağrılıyor
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
    
    return final_score

def save_model(model, file_path):
    print(f"Model {file_path} adresine kaydediliyor...")
    model_path = file_path.replace('.txt', f'_{config.MODEL_TYPE}.txt')
    joblib.dump(model, model_path)
    print(f"Model başarıyla {model_path} olarak kaydedildi.")