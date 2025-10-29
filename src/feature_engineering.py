# src/feature_engineering.py

import pandas as pd
import numpy as np
from tqdm import tqdm  # İşlemin ilerlemesini görmek için

def create_features(customers, history, reference_df):
    print("Veri sızıntısını önleyen, garantili özellik mühendisliği başlıyor...")
    
    # tqdm'ı pandas ile kullanmak için
    tqdm.pandas(desc="Özellikler Hesaplanıyor")

    # Temel veri hazırlığı
    history_copy = history.copy()
    history_copy['toplam_harcama'] = history_copy['mobile_eft_all_amt'].fillna(0) + history_copy['cc_transaction_all_amt'].fillna(0)
    
    # reference_df üzerinde her satır için özel özellikler hesaplayacağız
    # Bu fonksiyon, her (musteri, tarih) ikilisi için çalışacak
    def create_features_for_row(row):
        musteri_id = row['musteri_id']
        ref_dt = row['ref_dt']

        # Kural: Sadece referans tarihinden önceki işlem geçmişini al!
        relevant_history = history_copy[(history_copy['musteri_id'] == musteri_id) & (history_copy['ref_dt'] < ref_dt)]

        if relevant_history.empty:
            return pd.Series([0, 999, 0, 0, 0, 0, 0], index=[
                'aktif_ay_sayisi', 'gun_farki_son_islem', 'son_3ay_toplam_harcama',
                'son_6ay_toplam_harcama', 'son_3ay_ort_harcama', 'son_6ay_ort_harcama', 'trend_3ay_vs_6ay'
            ])

        # 1. Recency (Yenilik) - En Güçlü Sinyal
        son_islem_tarihi = relevant_history['ref_dt'].max()
        gun_farki_son_islem = (ref_dt - son_islem_tarihi).days

        # 2. Frequency (Sıklık)
        aktif_ay_sayisi = relevant_history['ref_dt'].nunique()

        # 3. Monetary & Trend (Parasal Değer ve Eğilim)
        son_3ay_mask = relevant_history['ref_dt'] >= (ref_dt - pd.DateOffset(months=3))
        son_6ay_mask = relevant_history['ref_dt'] >= (ref_dt - pd.DateOffset(months=6))

        son_3ay_toplam_harcama = relevant_history[son_3ay_mask]['toplam_harcama'].sum()
        son_6ay_toplam_harcama = relevant_history[son_6ay_mask]['toplam_harcama'].sum()
        
        son_3ay_ort_harcama = son_3ay_toplam_harcama / 3
        son_6ay_ort_harcama = son_6ay_toplam_harcama / 6

        # Trend: Son 3 ayın ortalaması, son 6 ayın ortalamasına göre nasıl?
        trend_3ay_vs_6ay = son_3ay_ort_harcama / (son_6ay_ort_harcama + 1e-6)

        return pd.Series([
            aktif_ay_sayisi, gun_farki_son_islem, son_3ay_toplam_harcama,
            son_6ay_toplam_harcama, son_3ay_ort_harcama, son_6ay_ort_harcama, trend_3ay_vs_6ay
        ], index=[
            'aktif_ay_sayisi', 'gun_farki_son_islem', 'son_3ay_toplam_harcama',
            'son_6ay_toplam_harcama', 'son_3ay_ort_harcama', 'son_6ay_ort_harcama', 'trend_3ay_vs_6ay'
        ])

    # .progress_apply() ile her satır için fonksiyonu çalıştır ve ilerlemeyi izle
    # Bu işlem biraz zaman alabilir ama DOĞRU sonucu verecektir.
    new_features = reference_df.progress_apply(create_features_for_row, axis=1)
    
    df = pd.concat([reference_df, new_features], axis=1)
    
    # Demografik verileri ekle
    df = pd.merge(df, customers, on='musteri_id', how='left')
    df.fillna(0, inplace=True)
    
    # Kategorik Değişkenler
    categorical_cols = ['gender', 'religion', 'work_type', 'work_sector']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

    print("Garantili özellik mühendisliği tamamlandı.")
    return df