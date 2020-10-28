import numpy as np
import pandas as pd

df = pd.read_csv('data-sets/impedance-transpose.csv').dropna().reset_index(drop=True)
label = ['100_THP', '90_THP', '80_THP', '70_THP', '65_THP', '50_THP', '30_THP', '15_THP', '10_THP', '100_PHA',
         '90_PHA', '80_PHA', '70_PHA', '65_PHA', '50_PHA', '30_PHA', '15_PHA', '10_PHA']
x = df.drop(columns=['Frequency', 'Label']).to_numpy(dtype='float64')
y = df['Label'].to_numpy(dtype='float64')

for i in range(len(x) - 1):
    print("{std}".format(at=label[i], std=np.std(x[i], dtype=np.float32)))
