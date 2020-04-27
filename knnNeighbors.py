import numpy as np
import pandas as pd
import prepareData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


x_train, x_test, y_train, y_test = prepareData.get()

# performs feature scaling
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
