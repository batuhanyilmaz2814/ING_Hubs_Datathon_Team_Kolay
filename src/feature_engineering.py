# src/feature_engineering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_features(customers, history, reference_df):
    """
    Ham verilerden model için özellik matrisi oluşturur.
    """
    print("Özellik mühendisliği adımı başlıyor (ZAMAN SERİSİ DELTA ÖZELLİKLERİ)...")

    # Adım 1: History tablosundaki eksik değerleri 0 ile doldurma
    transaction_cols = ['mobile_eft_all_cnt', 'mobile_eft_all_amt', 'cc_transaction_all_amt', 'cc_transaction_all_cnt']
    history[transaction_cols] = history[transaction_cols].fillna(0)

    # --- YENİLİK: İlk İşlem Tarihini Bulma (Müşteri Yaşı için) ---
    first_transaction_date = history.groupby('musteri_id')['ref_dt'].min().reset_index()
    first_transaction_date.rename(columns={'ref_dt': 'ilk_islem_tarihi'}, inplace=True)
    history = pd.merge(history, first_transaction_date, on='musteri_id', how='left')

    # --- ZAMAN SERİSİ DELTA ÖZELLİKLERİNİN OLUŞTURULMASI ---
    history.sort_values(by=['musteri_id', 'ref_dt'], inplace=True)

    # Farkı (Delta) hesaplanacak sütunlar
    cols_to_delta = ['mobile_eft_all_amt', 'cc_transaction_all_cnt', 'active_product_category_nbr']

    for col in cols_to_delta:
        history[f'{col}_prev'] = history.groupby('musteri_id')[col].shift(1)
        history[f'{col}_delta'] = history[col] - history[f'{col}_prev']

    delta_cols = [col for col in history.columns if '_delta' in col]
    history[delta_cols] = history[delta_cols].fillna(0)

    delta_agg_dict = {col: ['sum', 'mean'] for col in delta_cols}
    history_delta_agg = history.groupby('musteri_id').agg(delta_agg_dict).reset_index()

    history_delta_agg.columns = ['_'.join(col).strip() for col in history_delta_agg.columns.values]
    history_delta_agg.rename(columns={'musteri_id_': 'musteri_id'}, inplace=True)

    # --- TEMEL ÖZELLİK OLUŞTURMA AKIŞI ---

    # Adım 2: Her müşteri için TÜM ZAMANLARDAKİ işlem geçmişini özetleme
    agg_dict_all = {
        'mobile_eft_all_cnt': ['sum', 'mean'],
        'mobile_eft_all_amt': ['sum', 'mean'],
        'cc_transaction_all_amt': ['sum', 'mean'],
        'cc_transaction_all_cnt': ['sum', 'mean'],
        'active_product_category_nbr': ['mean', 'std'],
        'ref_dt': ['max', 'count'],
        'ilk_islem_tarihi': 'first'  # İlk işlem tarihini koru
    }

    history_agg_all = history.groupby('musteri_id').agg(agg_dict_all).reset_index()

    # Çoklu indeks sütun adlarını düzeltme
    history_agg_all.columns = ['_'.join(col).strip() for col in history_agg_all.columns.values]
    history_agg_all.rename(
        columns={'musteri_id_': 'musteri_id',
                 'ref_dt_max': 'son_islem_tarihi',
                 'ref_dt_count': 'aktif_ay_sayisi',
                 'ilk_islem_tarihi_first': 'ilk_islem_tarihi'},  # Yeni sütun adlandırması
        inplace=True)
    history_agg_all['ilk_islem_tarihi'] = pd.to_datetime(history_agg_all['ilk_islem_tarihi'])  # Tekrar datetime yap

    # Adım 3: Zamana Dayalı (Temporal) Özellikler Türetme (Uzun kod satırları basitleştirildi)
    df_temporal_base = pd.merge(reference_df, history, on='musteri_id', how='left', suffixes=('_ref', '_hist'))
    df_temporal_base = df_temporal_base[df_temporal_base['ref_dt_hist'] < df_temporal_base['ref_dt_ref']]

    new_transaction_cols = ['mobile_eft_all_cnt', 'mobile_eft_all_amt', 'cc_transaction_all_amt',
                            'cc_transaction_all_cnt']

    def create_temporal_features(df, window_months):
        df['days_diff'] = (df['ref_dt_ref'] - df['ref_dt_hist']).dt.days
        df_window = df[df['days_diff'] <= (window_months * 30.5)]

        agg_dict_window = {col: ['sum', 'mean'] for col in new_transaction_cols}

        agg_window = df_window.groupby('musteri_id').agg(agg_dict_window).reset_index()
        agg_window.columns = ['musteri_id'] + [f'{col}_{agg}_{window_months}m' for col, agg in agg_window.columns[1:]]

        if window_months == 1:
            agg_window[f'is_active_{window_months}m'] = agg_window[agg_window.columns[1]].apply(
                lambda x: 1 if x > 0 else 0)

        return agg_window

    agg_1m = create_temporal_features(df_temporal_base.copy(), 1)
    agg_3m = create_temporal_features(df_temporal_base.copy(), 3)
    agg_6m = create_temporal_features(df_temporal_base.copy(), 6)

    # Adım 4: Özellik Birleştirme ve Dönüşümler
    final_df = reference_df.copy()

    final_df = pd.merge(final_df, history_agg_all.drop('ilk_islem_tarihi', axis=1, errors='ignore'), on='musteri_id',
                        how='left')
    final_df = pd.merge(final_df, history_delta_agg, on='musteri_id', how='left')
    final_df = pd.merge(final_df, agg_1m, on='musteri_id', how='left')
    final_df = pd.merge(final_df, agg_3m, on='musteri_id', how='left')
    final_df = pd.merge(final_df, agg_6m, on='musteri_id', how='left')

    final_df['son_islem_gun_farki'] = (final_df['ref_dt'] - final_df['son_islem_tarihi']).dt.days
    final_df.drop('son_islem_tarihi', axis=1, inplace=True)

    # YENİ ÖZELLİK: Müşteri Yaşı
    final_df = pd.merge(final_df, history_agg_all[['musteri_id', 'ilk_islem_tarihi']], on='musteri_id', how='left')
    final_df['musteri_yasi_gun'] = (final_df['ref_dt'] - final_df['ilk_islem_tarihi']).dt.days
    final_df.drop('ilk_islem_tarihi', axis=1, inplace=True)

    final_df = pd.merge(final_df, customers, on='musteri_id', how='left')

    # --- RADİKAL DÖNÜŞÜM VE RANK ÖZELLİKLERİ ---
    epsilon = 1e-6

    # Logaritmik Dönüşüm
    amount_cols = [col for col in final_df.columns if 'amt' in col and ('sum' in col or 'mean' in col)]
    for col in amount_cols:
        final_df[f'log_{col}'] = np.log1p(final_df[col])

    # Sıra (Rank) Özellikleri
    rank_cols = ['cc_transaction_all_amt_sum', 'mobile_eft_all_amt_sum', 'aktif_ay_sayisi',
                 'musteri_yasi_gun']  # Yeni eklendi
    for col in rank_cols:
        final_df[f'rank_{col}'] = final_df[col].rank(method='dense', pct=True)

    # İŞLEM HACMİ KATEGORİZASYONU (BINNING)
    total_transaction_amount = final_df['mobile_eft_all_amt_sum'] + final_df['cc_transaction_all_amt_sum']
    bins = [0, total_transaction_amount.quantile(0.5),
            total_transaction_amount.quantile(0.75),
            total_transaction_amount.quantile(0.90),
            total_transaction_amount.max() + epsilon]
    labels = ['Low', 'Medium', 'High', 'VeryHigh']
    final_df['hacim_segmenti'] = pd.cut(total_transaction_amount,
                                        bins=bins, labels=labels, right=False, duplicates='drop').astype(object)

    # ORAN ÖZELLİKLERİ
    total_amount = total_transaction_amount + epsilon
    total_count = final_df['mobile_eft_all_cnt_sum'] + final_df['cc_transaction_all_cnt_sum'] + epsilon

    final_df['ratio_1m_to_all_amt'] = (final_df['mobile_eft_all_amt_sum_1m'] + final_df[
        'cc_transaction_all_amt_sum_1m']) / total_amount
    final_df['ratio_3m_to_all_amt'] = (final_df['mobile_eft_all_amt_sum_3m'] + final_df[
        'cc_transaction_all_amt_sum_3m']) / total_amount
    final_df['aktivite_degisim_1m_3m_cnt'] = final_df['mobile_eft_all_cnt_sum_1m'] / (
            final_df['mobile_eft_all_cnt_sum_3m'] + epsilon)
    final_df['eft_vs_cc_denge'] = (final_df['mobile_eft_all_amt_sum'] - final_df[
        'cc_transaction_all_amt_sum']) / total_amount
    final_df['cc_amt_per_active_month'] = final_df['cc_transaction_all_amt_sum'] / (
            final_df['aktif_ay_sayisi'] + epsilon)
    final_df['eft_amt_per_active_month'] = final_df['mobile_eft_all_amt_sum'] / (final_df['aktif_ay_sayisi'] + epsilon)
    final_df['eft_cc_count_ratio'] = final_df['mobile_eft_all_cnt_sum'] / (
            final_df['cc_transaction_all_cnt_sum'] + epsilon)

    # YENİ ORAN: Ortalama Bilet Boyutu
    final_df['avg_ticket_size'] = total_amount / total_count

    # NİHAİ NaN DOLDURMA
    cols_to_fill_with_missing = ['work_type', 'hacim_segmenti', 'gender', 'religion', 'work_sector']
    for col in cols_to_fill_with_missing:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna('Bilinmiyor')

    numeric_cols = final_df.select_dtypes(include=np.number).columns
    final_df[numeric_cols] = final_df[numeric_cols].fillna(0)

    # Adım 5: Kategorik değişkenleri işleme (KARDİNALİTE KONTROLÜ)
    MIN_COUNT = 100
    work_type_counts = final_df['work_type'].value_counts()
    frequent_work_types = work_type_counts[work_type_counts >= MIN_COUNT].index

    final_df['work_type_limited'] = np.where(
        final_df['work_type'].isin(frequent_work_types),
        final_df['work_type'],
        'Diğer_WorkType'
    )

    categorical_cols_to_use = ['work_type_limited', 'hacim_segmenti']

    for col in categorical_cols_to_use:
        final_df[col] = final_df[col].astype('category')

    cols_to_drop_final = ['gender', 'religion', 'work_sector', 'work_type']
    existing_cols_to_drop = [col for col in cols_to_drop_final if col in final_df.columns]
    final_df.drop(existing_cols_to_drop, axis=1, inplace=True)

    df = final_df

    print("Özellik mühendisliği tamamlandı.")
    return df