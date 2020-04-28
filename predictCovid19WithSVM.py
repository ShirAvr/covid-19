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

rowsCount, colCount = df_patients.shape
print("rows count after shape: ", rowsCount)
#print(df_patients.describe())
#print(df_patients.dtypes)

# seperate the depended param from the independed data
df_inputs = df_patients.drop(['corona_result', 'test_indication'], axis=1)

print(df_inputs.head())
print(df_inputs.tail())

x = df_inputs.values 	# inputs
y = df_patients.iloc[:, 6].values # outputs

print("=====")
print(x)
print(y)
print("=====")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=109)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("=====")
print(confusion_matrix(y_test, y_pred))
print("=====")
print(classification_report(y_test, y_pred))
print("=====")
print(balanced_accuracy_score(y_test, y_pred))
print("=====")

