import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

df = pd.read_csv('data-sets/three-stat.csv').dropna()\
    .reset_index(drop=True)

x = df.drop(columns=['S No.', 'Value', 'Label'])
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=False, test_size=0.4)

clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
clf_output = clf.predict(x_test)

print(confusion_matrix(y_test, clf_output))

print("SVM Precision score:{}".format(precision_score(y_test, clf_output)))
print("SVM Accuracy score:{}".format(accuracy_score(y_test, clf_output)))
