import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_train = pd.read_csv('data-sets/sensor_data_cancer_train.csv')
df_test = pd.read_csv('data-sets/sensor_data_cancer_test.csv')

# cancerous cell
data_thp_100 = df_train['100%_THP']
data_thp_80 = df_train['80%_THP']
data_thp_60 = df_train['60%_THP']
data_thp_50 = df_train['50%_THP']
data_thp_40 = df_train['40%_THP']
data_thp_30 = df_train['30%_THP']
data_thp_20 = df_train['20%_THP']
data_thp_10 = df_train['10%_THP']

# non-cancerous cell
data_pha = df_train['PHA(-)']
data_air = df_train['Air']

# test data
data_got = df_test['datagot']


data_train = np.array([data_thp_100, data_thp_80, data_thp_60, data_thp_50, data_thp_40, data_thp_30, data_thp_20,
                      data_thp_10, data_pha, data_air])

#plt.plot(data_thp_100)
# plt.plot(data_thp_80)
# plt.plot(data_thp_60)
plt.plot(data_thp_60, label="60")

plt.plot(data_thp_40, label="40")
plt.legend(loc="upper right")
plt.show()
