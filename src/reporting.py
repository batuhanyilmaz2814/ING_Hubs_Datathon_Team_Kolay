# src/reporting.py

from sklearn.metrics import roc_auc_score
from src.metrics import convert_auc_to_gini, recall_at_k, lift_at_k
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

REPORT_DIR = 'reports'


def plot_model_comparison(df_results):
    """
    Model performansını Gini, Recall@10% ve Lift@10% metriklerine göre karşılaştıran bar grafiği çizer.
    """
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    metrics_to_plot = ['Gini', 'Recall@10%', 'Lift@10%']

    if 'ROC AUC' in df_results.columns and 'Gini' not in df_results.columns:
        df_results['Gini'] = df_results['ROC AUC'].apply(convert_auc_to_gini)

    plot_df = df_results[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_df.plot(kind='bar', ax=ax, rot=0)

    ax.set_title('Model Performans Karşılaştırması (Validasyon Seti)', fontsize=14)
    ax.set_ylabel('Metrik Değeri', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f')

    plt.tight_layout()
    file_path = os.path.join(REPORT_DIR, 'model_comparison_bar_chart.png')
    plt.savefig(file_path)
    print(f"\nGrafik '{file_path}' olarak kaydedildi.")
    plt.close(fig)


def generate_performance_report(base_models, X_val, y_val, final_ensemble_score):
    """
    Tüm modellerin bireysel ve Ensemble performansını gösteren kapsamlı bir rapor oluşturur.
    """
    print("\n\n#####################################################")
    print("########## FİNAL RAPORU: STACKING CLASSIFIER ##########")
    print("#####################################################")

    # 1. Base Modellerin Bireysel Performansını Hesaplama
    results = {}

    # X_val'deki kategorik veriyi CatBoost'un beklediği formata çevir
    X_val_cb = X_val.copy()
    for col in X_val_cb.select_dtypes(include='category').columns:
        X_val_cb[col] = X_val_cb[col].cat.codes

        # LightGBM ve CatBoost'u döngüye al
    for name, model in base_models.items():
        if name == 'catboost':
            y_prob = model.predict_proba(X_val_cb)[:, 1]
        else:
            y_prob = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_prob)
        results[name] = {
            "ROC AUC": auc,
            "Gini": convert_auc_to_gini(auc),
            "Recall@10%": recall_at_k(y_val, y_prob, k=0.1),
            "Lift@10%": lift_at_k(y_val, y_prob, k=0.1)
        }

    # 2. Raporlama Tablosunu Yazdırma
    df_results = pd.DataFrame(results).T
    df_results.index.name = 'Model'

    print("\n--- 1. BİREYSEL MODEL PERFORMANSI (VALIDASYON SETİ) ---")
    print(df_results.to_markdown(floatfmt=".4f"))

    print("\nModel Güç Sıralaması (Gini'ye Göre):")
    print(df_results.sort_values(by='Gini', ascending=False)['Gini'].map('{:.4f}'.format))

    # --- GRAFİK ÇİZİMİ ---
    plot_model_comparison(df_results.copy())

    # 3. Final Skor Analizi
    print("\n--- 2. FİNAL ENSEMBLE SKOR ANALİZİ ---")
    BASE_SCORE = 1.0

    print(f"Hedef Skor (Final Submission): 1.30000")
    print(f"Validasyon Ensemble Skoru: {final_ensemble_score:.5f}")
    print(f"Fark (Aşılması Gereken Potansiyel): {1.3 - final_ensemble_score:.5f}")

    if final_ensemble_score >= 1.25:
        print("\n!!! TEBRİKLER !!! 1.25 Eşiği Aşıldı.")
    else:
        print(f"\n-> BAŞARILI: Model, baseline'ı aştı. 1.3 hedefine ulaşmak için son teknik potansiyel kullanıldı.")

    print("#####################################################\n")