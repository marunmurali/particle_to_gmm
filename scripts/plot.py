import numpy as np
import matplotlib.pyplot as plt

def tosecaxis(x): 
    return 10000.0 * x

def reversesecaxis(x): 
    return 0.0001 * x

methods = ["State Feedback with GMM", "State Feedback without GMM", "DWA with GMM"]

x_pos = np.arange(len(methods))
means = [0.004064800396, 0.00340337653, 0.006345127083]
stddev = [0.002558933224, 0.00178111465, 0.003211480271]
mean_times = [0.0001 * 81.00733027, 0.0001 * 73.05546622, 0.0001 * 92.82284513]

fig, ax = plt.subplots(constrained_layout=True, figsize=(9,6))

ax.bar(x_pos - 0.2, means, 0.4, yerr=stddev, align="center", capsize=5, color='r', label="mean square error")
ax.bar(x_pos + 0.2, mean_times, 0.4, color = 'b', label="travel time")

plt.xticks(x_pos, methods, rotation=60)

ax.legend()

ax.set_ylabel("Mean square error/$m^2$)")

secax_y = ax.secondary_yaxis('right', functions=(tosecaxis, reversesecaxis))

secax_y.set_ylabel("Average path following time/$s$")

ax.set_title('Comparison of error and travel time of 2 proposed methods and conventional method', fontsize = 14)

plt.show()
