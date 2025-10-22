# src/ensemble.py

import pandas as pd
from catboost import CatBoostClassifier
from src import config
import numpy as np
from sklearn.isotonic import IsotonicRegression
from src.metrics import ing_hubs_datathon_metric  # Yeni import


# Global sınıf ağırlığını hesaplayalım (CatBoost için)
def calculate_class_weight_cb(y):
    """ CatBoost için pozitif sınıf ağırlığını hesaplar (2x agresiflik). """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_factor = 2 * (neg_count / pos_count)
    return [1, scale_factor]


def train_catboost(X_train, y_train, categorical_features):
    """
    CatBoost modelini eğitir (Maliyet Duyarlı).
    """
    print("-> CatBoost modeli eğitiliyor (Maliyet Duyarlı)...")

    class_weights = calculate_class_weight_cb(y_train)

    cb_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': config.RANDOM_STATE,
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 5,
        'l2_leaf_reg': 3,
        'verbose': 0,
        'allow_writing_files': False,
        'class_weights': class_weights
    }

    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train, cat_features=categorical_features)

    print("-> CatBoost eğitimi tamamlandı.")
    return cb_model


def predict_ensemble(lgbm_model, catboost_model, X_test, weight_cb):
    """
    CatBoost ve LightGBM tahminlerini OPTİMİZE EDİLMİŞ ağırlıklı ortalama ile birleştirir.
    """
    lgbm_preds = lgbm_model.predict_proba(X_test)[:, 1]
    catboost_preds = catboost_model.predict_proba(X_test)[:, 1]

    final_preds = (weight_cb * catboost_preds) + ((1 - weight_cb) * lgbm_preds)

    return final_preds


def optimize_ensemble_weight(lgbm_preds, catboost_preds, y_true):
    """
    Validasyon tahminleri üzerinde CatBoost ağırlığını (0.00'dan 1.00'e kadar) optimize eder.
    """
    print("\n-> Ensemble Ağırlığı Optimizasyonu Başlıyor...")

    weights = np.arange(0.00, 1.01, 0.05)  # 0.00, 0.05, 0.10, ..., 1.00
    best_score = -1.0
    best_weight = 0.75  # Başlangıç varsayılan değer

    for w_cb in weights:
        w_lgbm = 1.0 - w_cb

        # Tahmini hesapla
        current_preds = (w_cb * catboost_preds) + (w_lgbm * lgbm_preds)

        # Skoru hesapla (Kalibrasyon olmadan metrik değeri)
        # Kalibrasyon, IsotonicRegression tarafından yapıldığı için, ağırlık optimizasyonunda
        # sadece temel modelin skorunu kullanmak daha güvenlidir.
        current_score = ing_hubs_datathon_metric(y_true, current_preds)

        if current_score > best_score:
            best_score = current_score
            best_weight = w_cb

    best_preds = (best_weight * catboost_preds) + ((1 - best_weight) * lgbm_preds)

    print(f"-> Optimizasyon Tamamlandı. En iyi skor ({best_score:.5f}) için CatBoost Ağırlığı: {best_weight:.2f}")
    return best_weight, best_preds


def calibrate_predictions(y_prob, y_true):
    """
    Izotonik Regresyon kullanarak olasılıkları kalibre eder.
    """
    print("-> Izotonik Kalibrasyon Modeli Eğitiliyor...")

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(np.asarray(y_prob), np.asarray(y_true))

    print("-> Kalibrasyon Eğitimi Tamamlandı.")
    return ir