import pandas as pd
import math
import matplotlib.pyplot as plt

epsilon_r = [20, 18, 16, 14, 12, 10]  # 10, 20, 40, 60, 80, 90
imp_0 = []
imp_1 = []
imp_2 = []
imp_3 = []
imp_4 = []
imp_5 = []
frequency = []

for freq in range(1, 100, 1):
    frequency.append(freq)
    imp_0.append((1 / (2 * math.pi * epsilon_r[0] * 4.12 * pow(10, -12) * freq * pow(10, 3))))
    imp_1.append((1 / (2 * math.pi * epsilon_r[1] * 4.12 * pow(10, -12) * freq * pow(10, 3))))
    imp_2.append((1 / (2 * math.pi * epsilon_r[2] * 4.12 * pow(10, -12) * freq * pow(10, 3))))
    imp_3.append((1 / (2 * math.pi * epsilon_r[3] * 4.12 * pow(10, -12) * freq * pow(10, 3))))
    imp_4.append((1 / (2 * math.pi * epsilon_r[4] * 4.12 * pow(10, -12) * freq * pow(10, 3))))
    imp_5.append((1 / (2 * math.pi * epsilon_r[5] * 4.12 * pow(10, -12) * freq * pow(10, 3))))

# save performance data
values = {'Frequency': frequency, '90_THP': imp_5, '80_THP': imp_4, '60_THP': imp_3, '40_THP': imp_2, '20_THP': imp_1,
          '10_THP': imp_0}
df_w = pd.DataFrame(values, columns=['Frequency', '90_THP', '80_THP', '60_THP', '40_THP', '20_THP', '10_THP'])
df_w.to_csv("data-sets/data-gen.csv", index=None, header=True)

plt.figure(1)
plt.plot(frequency, imp_0, '-b', linewidth=0.5, label='10 THP')
plt.plot(frequency, imp_1, '-g', linewidth=0.5, label='20 THP')
plt.plot(frequency, imp_2, '-r', linewidth=0.5, label='40 THP')
plt.plot(frequency, imp_3, '-c', linewidth=0.5, label='60 THP')
plt.plot(frequency, imp_4, '-m', linewidth=0.5, label='80 THP')
plt.plot(frequency, imp_5, '-y', linewidth=0.5, label='90 THP')
plt.xlabel('Frequency')
plt.ylabel('Impedance')
plt.legend(loc='best')
plt.grid()
plt.show()
