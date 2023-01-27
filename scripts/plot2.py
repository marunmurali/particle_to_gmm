import numpy as np
import matplotlib.pyplot as plt

# def tosecaxis(x): 
#     return 10000.0 * x

# def reversesecaxis(x): 
#     return 0.0001 * x

fig, ax = plt.subplots(constrained_layout=True, figsize=(10,6.5))

plt.rcParams.update({'font.size': 16})

methods = ["GMM_Flat", "GMM_squared", "GMM_max", "AMCL"]

x_pos = np.arange(len(methods))

means_1 = [0.00233936951, 0.001171748881, 0.002843341269, 0.002354102768]
stddev_1 = [0.001020981105, 0.0008282329555, 0.003358077133, 0.0003433720682]

means_2 = [0.007878163165, 0.007386000723, 0.0209865147, 0.01274732628]
stddev_2 = [0.004320072843, 0.005449611026, 0.01275619787, 0.004967927073]

ax.bar(x_pos - 0.2, means_1, 0.4, yerr=stddev_1, align="center", capsize=5, color='r', label="Setting 1 (lin=-0.1, ang=-0.5)")
ax.bar(x_pos + 0.2, means_2, 0.4, yerr=stddev_2, align="center", capsize=5, color='b', label="Setting 2 (lin=-0.1, ang=-0.1)")

plt.xticks(x_pos, methods, rotation=60, fontsize=16)
plt.yticks(fontsize=16)

ax.set_ylim([0, 0.05])

ax.set_axisbelow(True)
ax.grid(axis='y',linestyle='dotted', color='k')

ax.legend()

ax.set_ylabel("Mean square error/$m^2$", fontsize=16)

# secax_y = ax.secondary_yaxis('right', functions=(tosecaxis, reversesecaxis))

# secax_y.set_ylabel("Average path following time/$s$")

ax.set_title('An MSE comparison of 3 GMM-based controllers and a AMCL-based controller', fontsize=16)

plt.show()
