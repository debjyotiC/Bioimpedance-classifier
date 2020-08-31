import pandas as pd
import numpy as np
from sklearn import svm
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df_dataset = pd.read_csv('data-sets/data-gen.csv')

data_thp_100 = [np.max(df_dataset['100_THP']), np.min(df_dataset['100_THP']), np.average(df_dataset['100_THP'])]
data_thp_70 = [np.max(df_dataset['70_THP']), np.min(df_dataset['70_THP']), np.average(df_dataset['70_THP'])]

data_thp_30 = [np.max(df_dataset['30_THP']), np.min(df_dataset['30_THP']), np.average(df_dataset['30_THP'])]
data_thp_10 = [np.max(df_dataset['10_THP']), np.min(df_dataset['10_THP']), np.average(df_dataset['10_THP'])]

data_train = np.array([data_thp_100, data_thp_10], dtype=float)
data_label = np.array([0.0, 1.0], dtype=float)
data_test = np.array(data_thp_30, dtype=float)

clf = svm.SVC()
clf.fit(data_train, data_label)
print(clf.predict([data_test]))


