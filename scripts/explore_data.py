import pandas as pd

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.info())
print(df.isnull().sum())
print(df.head(10))
print(df.sample())
print(df["MultipleLines"].value_counts())
print(df["tenure"].value_counts(normalize=True) * 100)
print(df["tenure"].unique())

print(df.describe())
