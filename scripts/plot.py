import numpy as np
import matplotlib.pyplot as plt

def tosecaxis(x): 
    return 10000.0 * x

def reversesecaxis(x): 
    return 0.0001 * x

# methods = ["Conv", "SF_GMM_nm", "SF_GMM_sq", "DWA_GMM_nm", "DWA_GMM_sq"]

# x_pos = np.arange(len(methods))
# means = [0.00340, 0.00406, 0.00653, 0.01094, 0.00492]
# stddev = [0.00178, 0.00256, 0.00287, 0.00749, 0.00145]
# mean_times =[73.06 * 0.0001, 81.01 * 0.0001, 80.27 * 0.0001, 81.41 * 0.0001, 77.72 * 0.0001]

methods = ["SF_GMM_nm", "SF_GMM_sq", "DWA_GMM_nm", "DWA_GMM_sq"]

x_pos = np.arange(len(methods))
means = [0.00406, 0.00653, 0.01094, 0.00492]
stddev = [0.00256, 0.00287, 0.00749, 0.00145]
mean_times =[81.01 * 0.0001, 80.27 * 0.0001, 81.41 * 0.0001, 77.72 * 0.0001]

plt.rcParams.update({'font.size':  20})

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6.5))

ax.bar(x_pos - 0.2, means, 0.4, yerr=stddev, align="center", capsize=5, color='r', label="mean square error")
ax.bar(x_pos + 0.2, mean_times, 0.4, color = 'b', label="travel time")

plt.xticks(x_pos, methods, rotation=60, fontsize=20)
plt.yticks(fontsize=20)

ax.set_ylim([0, 0.02])
ax.legend()

ax.set_ylabel("Mean square error/$m^2$", fontsize=20)

secax_y = ax.secondary_yaxis('right', functions=(tosecaxis, reversesecaxis))

secax_y.set_ylabel("Average path following time/$s$", fontsize=20)

ax.set_title('Comparison of error MSE and travel time', fontsize = 22)

plt.show()
