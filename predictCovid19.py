import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

df_patients = pd.read_excel(open('corona_tested_individuals.xlsx', 'rb'), sheet_name='1 - tested person data') 
rowsCount, colCount = df_patients.shape
print(rowsCount)

df_patients = df_patients.dropna()
print(df_patients.dtypes)

df_patients['corona_result'] = np.where(df_patients['corona_result'] == 'שלילי', 0, 1)
df_patients['gender'] = np.where(df_patients['gender'] == 'זכר', 0, 1)
df_patients['age_60_and_above'] = np.where(df_patients['age_60_and_above'] == 'No', 0, 1)
df_patients['test_date'] = pd.to_datetime(df_patients.test_date)
df_patients['test_date'] = df_patients.test_date.apply(lambda x: x.toordinal())
df_patients['abroad'] = np.where(df_patients['test_indication'] == 'Abroad', 1, 0)
df_patients['metConfirmed'] = np.where(df_patients['test_indication'] == 'Contact with confirmed', 1, 0)

rowsCount, colCount = df_patients.shape
print(rowsCount)
print(df_patients.head())
print(df_patients.tail())
print(df_patients.describe())
print(df_patients.dtypes)

df_inputs = df_patients.drop(['corona_result', 'test_indication'], axis=1) # seperate the depended param from the independed data

print(df_inputs.head())
print(df_inputs.tail())

x = df_inputs.values 	# inputs
y = df_patients.iloc[:, 6].values # outputs

print("=====")
print(x)
print(y)
print("=====")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

knnClassifier = KNeighborsClassifier(n_neighbors=5)
knnClassifier.fit(x_train, y_train)
y_pred = knnClassifier.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
