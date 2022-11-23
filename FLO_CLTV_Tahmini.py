import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)

import warnings

warnings.filterwarnings("ignore")

df_ = pd.read_csv("my_work/flo_data_20k.csv")
df = df_.copy()

# Veriye Genel Bakış

df.info()
df.describe([0.01, 0.05, 0.25, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]).T
df.isnull().sum()


# ADIM 2: Ayırı değerleri tıraşla

# Aykırı değerler var, bu şekilde tahminler sağlıklı olmaz.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


v_list = ["order_num_total_ever_online",
          "order_num_total_ever_offline",
          "customer_value_total_ever_offline",
          "customer_value_total_ever_online"]

for col in v_list:
    replace_with_thresholds(df, col)

df.describe().T


# Total_order ve total_value değişkenlerini ekleyelim

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()
df.describe().T

# Bize verilen Tarih verilerinin tipi object, bunları datetime'a çevirelim.

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])
        # df[col] = df[col].astype("datetime64") olarak da yapılabilridi

df.dtypes

# today_date oluşturalım. Son sipariş tarihinden 2 gün sonrası için oluşturduk

df["last_order_date"].max()   # Timestamp('2021-05-30 00:00:00')

today_date = dt.datetime(2021, 6, 1)
type(today_date)

# RFM ile verilen değerler aynı şeyi ifade etmez
########## LİFE TİME VERİ YAPISI İÇİN DEĞERLERİ OLUŞTURALIM

# recency: Son sipariş üzerinden geçen zaman(Haftalık) =  last_order_date - first_order_date
# T: Müşteri yaşı(Haftalık) = today_date - first_order_date
# frequency: tekrar eden toplam satın alma sayısı(1'den fazla olmalıdır
# monetary: satın alma başına ortalama kazanç

######  burası olmadı   cltv_df = df.groupby("master_id").agg({"last_order_date": lambda x: (x - df["first_order_date"].min()).dt.days,
                                      # "first_order_date": lambda x: (df["last_order_date"].max() - x).dt.days,
                                      # "total_order": lambda x: x,
                                      # "total_value": lambda x: x})



cltv_df = df[["master_id", "last_order_date", "first_order_date", "total_order", "total_value"]]

cltv_df.head()

# istenilen değişiklikleri yapalım; master id kısmını customer_id yapalım

cltv_df.rename(columns={"master_id": "customer_id"}, inplace=True)

# recency_cltv_weekly adında bir değişken oluşturalım. Bunu lastorder-firstorder yapacağız

cltv_df["recency_cltv_weekly"] =  (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days

# "T_weekly" değişkenini oluşturalım today_date - firsdate

cltv_df["T_weekly"] = (today_date - cltv_df["first_order_date"]).dt.days
cltv_df["frequency"] = cltv_df["total_order"]

# monetary ortlaması için için yeni değişken oluşturalım

cltv_df["monetary_cltv_avg"] = cltv_df["total_value"] / cltv_df["frequency"]

cltv_df.head()

cltv_df = cltv_df.loc[:, ["customer_id","recency_cltv_weekly","T_weekly","frequency","monetary_cltv_avg"]]

cltv_df.head()

## ya da tek tek silinebilir

# del cltv_df["last_order_date"]
# del cltv_df["first_order_date"]
# del cltv_df["total_order"]
# del cltv_df["total_value"]

# frequency > 1 olmalı

cltv_df = cltv_df[cltv_df["frequency"] > 1]


# günler haftaya çevrilmeli

cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"] / 7
cltv_df["T_weekly"] = cltv_df["T_weekly"] / 7

cltv_df.head()

# biraz ilerledikten sonra indexte master_id olması gerektiğini gördüm

cltv_df.index = cltv_df["customer_id"]

del cltv_df["customer_id"]    # kalabalık olmaması için sildim

cltv_df.head()

####### BG-NBD modelinin kurulması ( satın alma sayısını modeller)

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# fitted with 19945 subjects, a: 2.13, alpha: 0.91, b: 6.90, r: 0.66>

# 1 hafta içinde satın alma beklediğimiz müşterilerin büyükten küçüğe sıralaması

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"]).sort_values(ascending=False)

# conditional_expected_number_of_purchases_up_to_time
# bunun kısa kullanımı ise "predict" dir

# 4 hafta içinde satın alma beklediğimiz müşterilerin büyükten küçüğe sıralaması

bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"]).sort_values(ascending=False)

## Bizden istenilen 3 ve 6 aylık satın alma tahmininini df'ye eklmememiz

# 3 ay

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

cltv_df.head()


# 6 ay

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

cltv_df.sort_values("exp_sales_6_month", ascending=False)

# tahmin sonuçlarının değerlendirilmesi

plot_period_transactions(bgf)
plt.show()

#### GammaGamma ( sipariş başına ortalama satın alma tutarını modeller)

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])



cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

# burada müşterinin ortlama kazandıracağı değer tahmini yapıldı. Herhangi bir süre için yapılmadı


cltv_df.sort_values("exp_average_value", ascending=False)


###################### CLTV TAHMİNİ

# 6 aylık CLTV hesaplyalım. CLTV değeri en yüksek 20 kişiye bakalım

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin zaman bilgisi(haftalık)
                                   discount_rate=0.01)
cltv

cltv = cltv.reset_index()

cltv.sort_values("clv", ascending=False).head(20)

# bunu cltv_df e yazdırmak istersek

cltv_final = cltv_df.merge(cltv, on="customer_id", how="left")

cltv_final.head()
# cltv_final = pd.merge(cltv_df, cltv)

# 6 aylık cltv ye göre 4 segmente bölelim

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values("clv", ascending=False).head(20)

cltv_final["segment"].value_counts()

cltv_final.groupby("segment").agg({"clv": ["mean", "min", "max"]})





