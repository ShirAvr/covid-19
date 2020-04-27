import numpy as np
import pandas as pd
import prepareData
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


x_train, x_test, y_train, y_test = prepareData.get()

# performs feature scaling
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

logreg = LogisticRegression()

# fit the model with data and training the model on the data
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
