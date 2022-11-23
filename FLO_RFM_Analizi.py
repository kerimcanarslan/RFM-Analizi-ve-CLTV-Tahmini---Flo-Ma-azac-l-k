
# İŞ PROBLEMİ:

# Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp
# bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak
# ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.


import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

# GÖREV 1

  #ADIM1

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.3f" %x)
df_ = pd.read_csv("my_work/flo_data_20k.csv")
df = df_.copy()


# ADIM 2

          # İlk 10 müşteri verisi

          df.head(10)
          df.columns
          df.describe().T

def check_df(dataframe, head=5):
    print("#"*9 + " SHAPE-BOYUT-ŞEKİL " + "#"*9)
    print(dataframe.shape)
    print("#"*9 + " DTYPE " + "#"*9)
    print(dataframe.dtypes)
    print("#" * 9 + " HEAD " + "#" * 9)
    print(dataframe.head(head))
    print("#" * 9 + " TAİL " + "#" * 9)
    print(dataframe.tail(head))
    print("#" * 9 + " NA-BOŞ DEĞER " + "#" * 9)
    print(dataframe.isnull().sum())
    print("#" * 9 + " QUANTİLES-NİCELLER " + "#" * 9)
    print(dataframe.describe().T)

check_df(df)


# grafikler

sns.countplot(x=df["order_channel"], data=df)
plt.show()

sns.histplot(x=df["customer_value_total_ever_online"])
plt.show(block=True)


sns.boxplot(x=df["customer_value_total_ever_online"])
plt.show(block=True)

df["customer_value_total_ever_online"].hist()
plt.show()


         df.isnull().values.any()
df.isnull().any()
         df.isnull().sum()
         df.dtypes
         df.head()

df["interested_in_categories_12"].value_counts()



# ADIM 3

# Her müşterinin toplam alışverisi sayısı "total_order"
# Her müşterinin toplam harcadığı para "total_value"

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# ADIM 4  Tarih ifade eden değişkenlerin tipini tarihe çevirme
df.dtypes

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])
        #df[col] = df[col].astype("datetime64[ns]")   # olarak da yapılabilridi

df.dtypes



# ADIM 5 Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

df["master_id"].nunique()    # Müşteri sayısı

df["order_channel"].value_counts()   # Hani kanalda kaç müşteri var

df.groupby("order_channel").agg({"total_order": ["count", "sum"],
                                 "master_id": "count",
                                 "total_value": ["sum", "mean"]})



#ADIM 6 En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df = df.sort_values("total_value", ascending=False)
df.head(10)

# Adım 7 En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df = df.sort_values("total_order", ascending=False)
df.head(10)


# Adım 8 Veri ön hazırlık sürecini fonksiyonlaştırınız.

def veri_on_hazırlık(dataframe):
    # ADIM 3

    # Her müşterinin toplam alışverisi sayısı "total_order"
    # Her müşterinin toplam harcadığı para "total_value"

    dataframe["total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_value"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    # ADIM 4  Tarih ifade eden değişkenlerin tipini tarihe çevirme

    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe




df.head()
df.dtypes

veri_on_hazırlık(df)

df.dtypes
df.head()

df.describe([0.01, 0.25, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]).T
#######################
# GÖREV 2: RFM metriklerinin hesaplanması

# RECENCY: Müşterinin son sipariş verdiği tarih ile analizin yapıldığı gün arasında geçen zaman, gün bazlı
# FREQUENCY: Müşterinin sipariş verme sıklığı, sipariş adeti bunu tanımlar
# MONETARY: Müşterinin siparişlerin toplam tutarı, firmaya kazandırdığı toplam para

df.head()

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)
type(today_date)


# Bonus Bilgi: type = dt.datetime(2021, 6, 1, format='%d/%m/%Y')


# ADIM 2

rfm = df.agg({"master_id": lambda x: x,
              "last_order_date": lambda x: (today_date - x).days,
              "total_order": lambda x: x,
              "total_value": lambda x: x})


rfm.columns = ["master_id", "recency", "frequency", "monetary"]

rfm.head()

rfm.describe().T


# GÖREV 3
######  RFM metrikleri ile RFM skorlarını oluşturma

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])

rfm["recency_score"].value_counts()
rfm["frequency_score"].value_counts()
rfm["monetary_score"].value_counts()

#rfm


rfm.head()

rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))



rfm[rfm["RF_SCORE"] == "55"]     # champ. müşteriler çıktı

rfm[rfm["RF_SCORE"] == "11"]     # en değersiz sınıf

# RF SINIFLANDIRMASI VE İSİMLER

# GÖREV 4

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm

rfm["RF_SCORE"].value_counts()

# GÖREV 5

 # Adım1

rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary":"mean"})

 # Adım2

rfm.head()

cat_df = df[["master_id", "interested_in_categories_12"]]

rfm = pd.merge(rfm, cat_df)
rfm.head(15)


womendf = rfm[["master_id", "segment", "interested_in_categories_12"]]


womendf = womendf.loc[(womendf["interested_in_categories_12"].str.contains("KADIN")) &
                      ((womendf["segment"] == "loyal_customers") | (womendf["segment"] == "champions"))]



womendf    # bu dataframe içinde loyal_customers ve champions sınıfında ve daha önce KADIN kategorisinde alışceriş yapmış kişilern İDsi var.

# Direkt oalrak bu dosyayı çıkarabiliriz, ya da sadece İD numaralarını alabiliriz

womendf[["master_id"]].to_csv("a_target_customer_id.csv")

# yaparak id ileri çıkarırırız.



# Adım 3


boys_40df = rfm[["master_id", "segment", "interested_in_categories_12"]]
boys_40df


boys_40df = boys_40df.loc[((boys_40df["interested_in_categories_12"].str.contains("COCUK")) |
                           (boys_40df["interested_in_categories_12"].str.contains("ERKEK"))) &
                           ((boys_40df["segment"] == "hibernating") |
                           (boys_40df["segment"] == "cant_loose") |
                           (boys_40df["segment"] == "new_customers"))]



boys_40df


