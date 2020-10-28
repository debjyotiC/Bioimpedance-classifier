import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

df_train = pd.read_csv('./data-sets/data-gen.csv')

df = pd.read_csv('data-sets/impedance-transpose.csv').dropna().reset_index(drop=True)
label = ['100_THP', '90_THP', '80_THP', '70_THP', '65_THP', '50_THP', '30_THP', '15_THP', '10_THP', '100_PHA',
         '90_PHA', '80_PHA', '70_PHA', '65_PHA', '50_PHA', '30_PHA', '15_PHA', '10_PHA']

x = df.drop(columns=['Frequency', 'Label']).to_numpy(dtype='float64')
y = df['Label'].to_numpy(dtype='float64')


def fit_func(freq, C, R):
    c_imp = 1 / (2 * np.pi * C * freq * pow(10, 3))
    return np.sqrt(pow(c_imp, 2))


params = curve_fit(fit_func, x, y)

[c, r] = params[0]

print("Coeff. C is {c} and R is {r}".format(c=c, r=r))

c_eq = 1 / (2 * np.pi * c * x * pow(10, 3))
z_eq = np.sqrt(pow(r, 2) + pow(c_eq, 2))

plt.plot(x, y, '--', color='red', label="data")
plt.plot(x, z_eq, '--', color='blue', label="optimized data")
plt.legend()
plt.show()
