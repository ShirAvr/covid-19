import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

df_patients = pd.read_excel(open('corona_tested_individuals.xlsx', 'rb'), sheet_name='1 - tested person data')
rowsCount, colCount = df_patients.shape
print("rows count before shape: ", rowsCount)


df_patients = df_patients.dropna()
print(df_patients.dtypes)

df_patients['corona_result'] = np.where(df_patients['corona_result'] == 'שלילי', 0, 1)
df_patients['gender'] = np.where(df_patients['gender'] == 'זכר', 0, 1)
df_patients['age_60_and_above'] = np.where(df_patients['age_60_and_above'] == 'No', 0, 1)
df_patients['test_date'] = pd.to_datetime(df_patients.test_date)
df_patients['test_date'] = df_patients.test_date.apply(lambda x: x.toordinal())
df_patients['abroad'] = np.where(df_patients['test_indication'] == 'Abroad', 1, 0)
df_patients['metConfirmed'] = np.where(df_patients['test_indication'] == 'Contact with confirmed', 1, 0)

print("positive cases: ")
print(df_patients[df_patients['corona_result'] == 1].shape)

print("positive cases with cough: ")
rowsCount, colCount = df_patients[(df_patients['cough'] == 1) & (df_patients['corona_result'] == 1)].shape
print(rowsCount)

print("positive cases with fever: ")
rowsCount, colCount = df_patients[(df_patients['fever'] == 1) & (df_patients['corona_result'] == 1)].shape
print(rowsCount)


print("positive cases with head_ache, cough, fever: ")
rowsCount, colCount = df_patients[(df_patients['head_ache'] == 1) & (df_patients['cough'] == 1) &
                                  (df_patients['fever'] == 1) & (df_patients['corona_result'] == 1)].shape
print(rowsCount)

print("negative cases with head_ache, cough, fever: ")
rowsCount, colCount = df_patients[(df_patients['head_ache'] == 1) & (df_patients['cough'] == 1) &
                                  (df_patients['fever'] == 1) & (df_patients['corona_result'] == 0)].shape
print(rowsCount)

print("positive tests above 60: ")
rowsCount, colCount = df_patients[(df_patients['corona_result'] == 1) & (df_patients['age_60_and_above'] == 1)].shape
print(rowsCount)

print("positive tests under 60: ")
rowsCount, colCount = df_patients[(df_patients['corona_result'] == 1) & (df_patients['age_60_and_above'] == 0)].shape
print(rowsCount)

print("tests above 60: ")
rowsCount, colCount = df_patients[(df_patients['age_60_and_above'] == 1)].shape
print(rowsCount)

print("tests under 60: ")
rowsCount, colCount = df_patients[(df_patients['age_60_and_above'] == 0)].shape
print(rowsCount)

