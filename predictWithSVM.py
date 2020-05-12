import numpy as np
import pandas as pd
import prepareData
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, cross_val_score


df_x, x, y = prepareData.get()

# performs feature scaling
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

accuracy_scores = []
balanced_accuracy_scores = []
total_x_test = []
total_y_test = []
total_y_pred = []

kfold = KFold(n_splits=10, random_state=100, shuffle=True)
for train_index, test_index in kfold.split(x):
  x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

  # fit the model with data and training the model on the data
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  total_x_test.extend(x_test)
  total_y_test.extend(y_test)
  total_y_pred.extend(y_pred)

  accuracy_scores.append(accuracy_score(y_test, y_pred))
  balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

print("===========accuracy scores==============")
print(accuracy_scores)
print("accuracy", np.mean(accuracy_scores))
print("balanced accuracy", np.mean(balanced_accuracy_scores))

print("===========results==============")
print(confusion_matrix(total_y_test, total_y_pred))
print(classification_report(total_y_test, total_y_pred))
print(balanced_accuracy_score(total_y_test, total_y_pred))



