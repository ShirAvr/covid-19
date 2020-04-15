import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.cluster import KMeans
df_line_list = pd.read_csv("novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
df_country_info = pd.read_csv("datasets/covid19countryinfo.csv")

df_line_list = df_line_list.filter(["id", "reporting date", "country", "gender", "age", "death", "recovered"]) 
df_line_list = df_line_list.dropna()
rowsCount, colCount = df_line_list.shape
print(rowsCount) 

print(df_line_list.head())
print(df_line_list.tail())

print(df_line_list.describe())

print("========")
df_country_info = df_country_info.filter(["country", "test", "testpop", "quarantine", "schools", "publicplace", "gatheringlimit"]) 
countryRowsCount, _ = df_country_info.shape
print(countryRowsCount)
print(df_country_info.head())