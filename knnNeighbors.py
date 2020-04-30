import numpy as np
import pandas as pd
import prepareData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt 
import seaborn as sns


x, y = prepareData.get()

# performs feature scaling
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

knnClassifier = KNeighborsClassifier(n_neighbors=25)

accuracy_scores = []
balanced_accuracy_scores = []
total_x_test = []
total_y_test = []
total_y_pred = []

kfold = KFold(n_splits=10, random_state=100, shuffle=True)
for train_index, test_index in kfold.split(x):
  x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

  # fit the model with data and training the model on the data
  knnClassifier.fit(x_train, y_train)
  y_pred = knnClassifier.predict(x_test)
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

### Visualize accuracy for each iteration
df_scores = pd.DataFrame(accuracy_scores, columns=['Scores'])
sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y="Scores", data=df_scores)
plt.show()
sns.set()

y_scores = knnClassifier.predict_proba(total_x_test)
fpr, tpr, threshold = roc_curve(total_y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, 'b', label = 'AUC (area = %0.2f)' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for knn')
plt.savefig('knn_roc')
plt.show()
