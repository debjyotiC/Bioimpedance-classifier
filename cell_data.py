import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

volume_percentage = [1, 0.9, 0.8, 0.7, 0.65, 0.5, 0.3, 0.15, 0.1]
radius_cell = [3.1, 8]  # normal, thp-1
imp_normal_100 = []
imp_normal_90 = []
imp_normal_80 = []
imp_normal_70 = []
imp_normal_65 = []
imp_normal_50 = []
imp_normal_30 = []
imp_normal_15 = []
imp_normal_10 = []


imp_thp_100 = []
imp_thp_90 = []
imp_thp_80 = []
imp_thp_70 = []
imp_thp_65 = []
imp_thp_50 = []
imp_thp_30 = []
imp_thp_15 = []
imp_thp_10 = []


frequency = []


def epsilon_mix(fraction, radius):
    phi = (100 * fraction * pow(10, 6) * 4 / 3 * math.pi * radius ** 3) / (400 * pow(10, -6) * pow(10, -3))
    e_mix = 80 * (1 + 2 * phi * (-0.49))/(1 - phi * (-0.49))
    return e_mix


def sensor_cap(epsilon_r, f):
    cap = 8.85 * pow(10, -12) * epsilon_r * 1160 * pow(10, -6)
    impedance = pow((2 * math.pi * f * pow(10, 3) * 400 * cap), -1)
    return impedance


for freq in range(1, 101, 1):
    frequency.append(freq)
    imp_normal_100.append(sensor_cap(epsilon_mix(volume_percentage[0], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_90.append(sensor_cap(epsilon_mix(volume_percentage[1], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_80.append(sensor_cap(epsilon_mix(volume_percentage[2], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_70.append(sensor_cap(epsilon_mix(volume_percentage[3], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_65.append(sensor_cap(epsilon_mix(volume_percentage[4], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_50.append(sensor_cap(epsilon_mix(volume_percentage[5], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_30.append(sensor_cap(epsilon_mix(volume_percentage[6], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_15.append(sensor_cap(epsilon_mix(volume_percentage[7], radius_cell[0] * pow(10, -6)), freq))
    imp_normal_10.append(sensor_cap(epsilon_mix(volume_percentage[8], radius_cell[0] * pow(10, -6)), freq))

    imp_thp_100.append(sensor_cap(epsilon_mix(volume_percentage[0], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_90.append(sensor_cap(epsilon_mix(volume_percentage[1], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_80.append(sensor_cap(epsilon_mix(volume_percentage[2], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_70.append(sensor_cap(epsilon_mix(volume_percentage[3], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_65.append(sensor_cap(epsilon_mix(volume_percentage[4], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_50.append(sensor_cap(epsilon_mix(volume_percentage[5], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_30.append(sensor_cap(epsilon_mix(volume_percentage[6], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_15.append(sensor_cap(epsilon_mix(volume_percentage[7], radius_cell[1] * pow(10, -6)), freq))
    imp_thp_10.append(sensor_cap(epsilon_mix(volume_percentage[8], radius_cell[1] * pow(10, -6)), freq))


# save performance data
values = {'Frequency': frequency, '100_THP': imp_thp_100, '90_THP': imp_thp_90, '80_THP': imp_thp_80,
          '70_THP': imp_thp_70, '65_THP': imp_thp_65, '50_THP': imp_thp_50, '30_THP': imp_thp_30, '15_THP': imp_thp_15,
          '10_THP': imp_thp_10, '100_PHA': imp_normal_100, '90_PHA': imp_normal_90, '80_PHA': imp_normal_80, '70_PHA': imp_normal_70,
          '65_PHA': imp_normal_65, '50_PHA': imp_normal_50, '30_PHA': imp_normal_30, '15_PHA': imp_normal_15,
          '10_PHA': imp_normal_10}
df_w = pd.DataFrame(values, columns=['Frequency', '100_THP', '90_THP', '80_THP', '70_THP', '65_THP', '50_THP',
                                     '30_THP', '15_THP', '10_THP', '100_PHA', '90_PHA', '80_PHA', '70_PHA', '65_PHA', '50_PHA',
                                     '30_PHA', '15_PHA', '10_PHA'])
df_w.to_csv("data-sets/data-gen.csv", index=None, header=True)


# plt.figure(1)
# plt.plot(frequency, imp_normal, '-k', linewidth=1, label='100 Normal')
# plt.plot(frequency, imp_thp_100, '-b', linewidth=1, label='100 THP')
# plt.plot(frequency, imp_thp_90, '-g', linewidth=1, label='90 THP')
# plt.plot(frequency, imp_thp_70, '-r', linewidth=1, label='70 THP')
# plt.plot(frequency, imp_thp_65, '-c', linewidth=1, label='65 THP')
# plt.plot(frequency, imp_thp_50, '-c', linewidth=1, label='50 THP')
# plt.plot(frequency, imp_thp_30, '-m', linewidth=1, label='30 THP')
# plt.plot(frequency, imp_thp_15, '-c', linewidth=1, label='15 THP')
# plt.plot(frequency, imp_thp_10, '-y', linewidth=1, label='10 THP')
# plt.xlabel('Frequency')
# plt.ylabel('Impedance')
# plt.legend(loc='best')
# plt.grid()
# plt.show()

