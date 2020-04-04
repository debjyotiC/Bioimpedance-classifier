import pandas as pd
import os
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df = pd.read_csv('data-sets/sensor_data_cancer_train_2.csv')
properties = list(df.columns.values)


print(properties)