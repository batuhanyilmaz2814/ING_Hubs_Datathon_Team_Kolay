# src/model_training.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src import config
import joblib
import optuna
from optuna.integration import LightGBMPruningCallback
import numpy as np  # Gerekli

# Yeni metrik modülümüzü ve içindeki fonksiyonları import edelim
from src.metrics import ing_hubs_datathon_metric, recall_at_k, lift_at_k, convert_auc_to_gini


def prepare_data_for_modeling(df):
    """
    Özellik matrisini modellemeye hazır hale getirir (X ve y olarak ayırır).
    """
    features_to_drop = [config.TARGET_COLUMN, 'musteri_id', 'ref_dt', 'province']
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]

    X = df.drop(columns=existing_cols_to_drop)
    y = df[config.TARGET_COLUMN]

    categorical_features = X.select_dtypes(include='category').columns.tolist()

    return X, y, categorical_features


def split_data(X, y, categorical_features):
    """
    Veriyi eğitim ve validasyon setlerine ayırır.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Validasyon seti boyutu: {X_val.shape}")

    return X_train, X_val, y_train, y_val, categorical_features


# --- OPTUNA FONKSİYONU ---
def objective(trial, X_train, y_train, X_val, y_val, categorical_features):
    """
    Optuna'nın optimize edeceği amaç fonksiyonu.
    Hedef: Validasyon AUC'yi maksimize etmek.
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 16, 32),
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'n_jobs': -1,
        'random_state': config.RANDOM_STATE,
        'verbose': -1
    }

    pruning_callback = LightGBMPruningCallback(trial, 'auc', valid_name='valid_0')

    model = lgb.LGBMClassifier(**params)

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False), pruning_callback],
              categorical_feature=categorical_features if categorical_features else 'auto')

    return model.best_score_['valid_0']['auc']


def optimize_hyperparameters(X, y, categorical_features):
    """
    Optuna optimizasyonunu çalıştırır ve en iyi parametreleri döndürür.
    """
    print("\n--- PARAMETRE OPTİMİZASYONU BAŞLIYOR (OPTUNA İLE) ---")

    # Optimizasyon için X'i ayır
    X_train, X_val, y_train, y_val, _ = split_data(X, y, categorical_features)

    # Optuna Study oluştur
    study = optuna.create_study(direction='maximize')

    # Optimizasyonu çalıştır (Örnek olarak 30 deneme)
    N_TRIALS = 30
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, categorical_features),
                   n_trials=N_TRIALS)

    print(f"Optimizasyon Tamamlandı. En iyi AUC: {study.best_value:.5f}")

    # En iyi parametreleri al ve temel parametreleri ekle
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['boosting_type'] = 'gbdt'
    best_params['n_estimators'] = 5000

    return best_params


def train_lgbm_with_early_stopping(X_train, y_train, X_val, y_val, params, categorical_features):
    """
    Erken Durdurma mekanizması ile LightGBM modelini eğitir ve en iyi iterasyon sayısını döndürür.
    """
    print("-> LightGBM Modeli Erken Durdurma ile Eğitiliyor...")

    model_params = params.copy()

    model = lgb.LGBMClassifier(**model_params, random_state=config.RANDOM_STATE)

    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=callbacks,
              categorical_feature=categorical_features if categorical_features else 'auto')

    best_n_estimators = model.best_iteration_

    print(f"-> LightGBM Eğitimi Tamamlandı. En iyi iterasyon sayısı: {best_n_estimators}")
    return model, best_n_estimators


def train_final_model_full_data(X, y, best_params, categorical_features=None, n_estimators=None):
    """
    Bulunan en iyi parametrelerle (ve en iyi tur sayısıyla) nihai modeli TÜM VERİ üzerinde eğitir.
    """
    print("\nNihai model TÜM VERİ üzerinde eğitiliyor (LightGBM)...")

    model_params = best_params.copy()
    if n_estimators is not None:
        model_params['n_estimators'] = n_estimators

    final_model = lgb.LGBMClassifier(**model_params, random_state=config.RANDOM_STATE)

    final_model.fit(X, y,
                    categorical_feature=categorical_features if categorical_features else 'auto')

    print("Nihai model eğitimi tamamlandı.")
    return final_model


def save_model(model, file_path):
    """
    Eğitilmiş modeli diske kaydeder.
    """
    print(f"Model {file_path} adresine kaydediliyor...")
    joblib.dump(model, file_path)
    print("Model başarıyla kaydedildi.")