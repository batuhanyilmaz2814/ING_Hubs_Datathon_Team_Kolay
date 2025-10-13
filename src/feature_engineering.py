# src/feature_engineering.py

import pandas as pd

def create_features(customers, history, reference_df):
    """
    Ham verilerden model için özellik matrisi oluşturur.

    Süreç:
    1. İşlem geçmişindeki (history) eksik değerleri 0 ile doldurur.
    2. Her müşteri için referans tarihinden önceki tüm işlemleri özetler (toplam, ortalama, vb.).
    3. Müşterinin ne kadar süredir aktif olduğunu ve son işleminin ne zaman olduğunu hesaplar.
    4. Bu özetlenmiş özellikleri demografik verilerle birleştirir.
    5. Kategorik değişkenleri modelin anlayacağı formata (one-hot encoding) çevirir.
    """
    print("Özellik mühendisliği adımı başlıyor...")

    # Adım 1: History tablosundaki eksik değerleri 0 ile doldurma
    transaction_cols = ['mobile_eft_all_cnt', 'mobile_eft_all_amt', 'cc_transaction_all_amt', 'cc_transaction_all_cnt']
    history[transaction_cols] = history[transaction_cols].fillna(0)

    # Adım 2: Her müşteri için işlem geçmişini özetleme
    # Bu adımda tüm zamanlardaki işlem verilerini toplayıp özetliyoruz.
    # Daha karmaşık özellikler (örn: son 3 ay) ileride eklenebilir.
    agg_dict = {
        'mobile_eft_all_cnt': ['sum', 'mean'],
        'mobile_eft_all_amt': ['sum', 'mean'],
        'cc_transaction_all_amt': ['sum', 'mean'],
        'cc_transaction_all_cnt': ['sum', 'mean'],
        'active_product_category_nbr': ['mean', 'std'],
        'ref_dt': ['max', 'count'] # 'max' son işlem tarihini, 'count' aktif ay sayısını verir
    }

    history_agg = history.groupby('musteri_id').agg(agg_dict).reset_index()

    # Sütun isimlerini daha anlaşılır hale getirelim (örn: ('sum', 'mobile_eft_all_cnt') -> 'sum_mobile_eft_all_cnt')
    history_agg.columns = ['_'.join(col).strip() for col in history_agg.columns.values]
    history_agg.rename(columns={'musteri_id_': 'musteri_id', 'ref_dt_max': 'son_islem_tarihi', 'ref_dt_count': 'aktif_ay_sayisi'}, inplace=True)
    
    # Adım 3: Referans tablosu ile birleştirme ve yeni zaman özellikleri türetme
    # reference_df -> df_ref_train veya df_ref_test olabilir.
    df = pd.merge(reference_df, history_agg, on='musteri_id', how='left')

    # Recency (Yenilik) özelliği: Referans tarihi ile son işlem tarihi arasındaki gün farkı
    df['son_islem_gun_farki'] = (df['ref_dt'] - df['son_islem_tarihi']).dt.days
    
    # Artık 'son_islem_tarihi' sütununa ihtiyacımız yok.
    df.drop('son_islem_tarihi', axis=1, inplace=True)

    # Adım 4: Demografik verileri birleştirme
    df = pd.merge(df, customers, on='musteri_id', how='left')

    # Adım 5: Kategorik değişkenleri işleme
    # work_sector'daki eksikliği "Bilinmiyor" kategorisi ile dolduralım.
    df['work_sector'] = df['work_sector'].fillna('Bilinmiyor')

    # One-Hot Encoding için kategorik sütunları belirleyelim.
    # province gibi çok fazla kategoriye sahip sütunları şimdilik dışarıda bırakıyoruz.
    categorical_cols = ['gender', 'religion', 'work_type', 'work_sector']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

    print("Özellik mühendisliği tamamlandı.")
    print(df)
    return df