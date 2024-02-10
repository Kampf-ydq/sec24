import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm, gaussian_kde


# Samples
# Unit is length
flow_lgths = [6, 890, 3, 216, 328, 160, 526, 86, 1904, 12, 394, 548, 110, 1, 1345, 2, 344, 4, 64, 1174, 4, 834, 1006, 10, 20,
 292, 228, 370, 70, 16, 20, 10, 20, 194, 376, 70, 152, 94, 3190, 530, 228, 264, 150, 36, 150, 74, 12, 120, 28, 68,
 106, 506, 14, 268, 14, 66, 482, 48, 8, 184, 34, 84, 16, 28, 38, 74, 32, 68, 30, 8, 10, 30, 46, 8, 48, 20, 4, 18,
 304, 116, 738, 302, 284, 88, 436, 52, 114, 208, 40, 78, 940, 72, 62, 522, 400, 2, 136, 74, 8, 2, 100, 2, 10, 42,
 46, 66, 20, 66, 22, 18, 204, 164, 178, 152, 6, 34, 238, 158, 252, 934, 552, 154, 68, 34, 34, 8, 246, 120, 660, 400,
 14, 542, 236, 20, 114, 34, 2, 26, 18, 38, 10, 98, 40, 26, 80, 94, 178, 80, 68, 142, 320, 166, 148, 390, 686, 270,
 94, 576, 116, 28, 10, 6, 6, 8, 192, 62, 592, 6, 10, 122, 778, 196, 88, 558, 252, 350, 666, 644, 118, 498, 178, 6,
 4, 34, 4, 86, 340, 186, 392, 1972, 172, 334, 86, 210, 340, 114, 164, 68, 354, 52, 6, 12, 16, 294, 64, 38, 4, 4, 6,
 160, 252, 154, 542, 186, 728, 250, 164, 258, 402]
FLOW_SIZE = 500
short_flow = [sf for sf in flow_lgths if sf <= FLOW_SIZE]
long_flow = [lf for lf in flow_lgths if lf > FLOW_SIZE]

# Calculate the probability density function by gaussian_kde
"""
xmin, xmax = min(flow_lgths), max(flow_lgths)
x = np.linspace(xmin, xmax, 100)
kde = gaussian_kde(flow_lgths)
pdf = kde(x)
kde1 = gaussian_kde(short_flow)
pdf1 = kde1(x)
kde2 = gaussian_kde(long_flow)
pdf2 = kde2(x)

# Red:#FE8083   Purple:#8E8EFE   Green:#58D6A6  Gray:#828282   Yellow:#FF9F3A
plt.plot(x, pdf1, '#FE8083', label='Short', linewidth=1)
plt.fill_between(x, pdf1, alpha=0.8, color='#FE8083')

plt.plot(x, pdf2, '#58D6A6', label='Long', linewidth=1)
plt.fill_between(x, pdf2, alpha=0.5, color='#58D6A6')

plt.plot(x, pdf, '#8E8EFE', label='All', linewidth=1)
plt.fill_between(x, pdf,  alpha=0.8, color='#8E8EFE')

plt.title('PDF')
plt.xlabel('X-axis')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.ylabel('Probability Density')
"""


# Unit is second (/s)
all_time = [2.03, 444.328, 11.427, 107.111, 163.134, 79.099, 262.124, 42.067, 951.707, 618.176, 196.133, 273.244, 49.021, 0, 671.831, 0.01, 171.071, 3.762, 31.057, 586.258, 0.213, 416.326, 502.381, 55.354, 8.996, 145.162, 113.063, 184.178, 34.024, 7.011, 9.017, 4.002, 8.959, 96.054, 187.206, 34.023, 75.078, 46.024, 1595.067, 264.205, 113.042, 131.076, 74.033, 17.022, 74.043, 36.044, 5.011, 59.052, 13.033, 33.043, 52.032, 252.151, 6.011, 133.122, 6.033, 32.022, 240.135, 22.999, 2.973, 91.109, 16.005, 41.024, 7.012, 13.017, 18.026, 36.024, 14.984, 33.04, 13.981, 3.018, 4.015, 14.028, 22.03, 2.997, 23.02, 8.997, 1, 7.987, 151.004, 57.045, 368.103, 150.073, 141.056, 43.046, 217.084, 25.064, 56.054, 103.075, 19.022, 38.018, 469.322, 35.032, 30.022, 260.217, 199.099, 0.011, 67.04, 36.011, 2.993, 0.012, 48.997, 0.012, 3.996, 20.046, 21.973, 32.006, 9.008, 32.03, 10.008, 8.021, 101.085, 81.062, 88.077, 75.057, 2.012, 16.021, 118.077, 78.042, 125.1, 466.263, 275.141, 76.088, 33.033, 16.022, 16.02, 3.011, 122.088, 59.076, 329.114, 199.101, 6.023, 270.138, 117.055, 8.991, 56.05, 16, 0.012, 12.006, 7.974, 18.04, 4.007, 48.013, 18.992, 12.005, 39.027, 46.02, 88.048, 39.045, 33.023, 70.068, 159.076, 82.055, 73.045, 194.084, 342.272, 134.153, 46.031, 287.16, 57.053, 13.012, 4.021, 2.012, 2.011, 3.014, 95.051, 30.045, 295.166, 2.012, 4.011, 60.021, 388.235, 97.088, 43.034, 278.204, 125.087, 174.208, 332.296, 321.247, 58.07, 248.218, 88.077, 1.996, 1.002, 15.974, 0.995, 41.992, 169.117, 92.054, 195.052, 985.538, 85.033, 166.147, 42.056, 104.098, 169.053, 56.045, 81.084, 33.034, 176.078, 25.021, 1.996, 5.009, 6.998, 146.151, 31.016, 18.004, 0.987, 0.988, 1.993, 79.005, 125.097, 76.099, 270.162, 92.077, 363.083, 124.1, 81.051, 128.073, 200.176]
all_time = [aft for aft in all_time if aft >= 1]
short_time = [sft for sft in all_time if sft <= 200]
long_time = [lft for lft in all_time if lft > 200]
xmin, xmax = min(all_time), max(all_time)
x = np.linspace(xmin, xmax, 100)
kde = gaussian_kde(all_time)
pdf = kde(x)
kde1 = gaussian_kde(short_time)
pdf1 = kde1(x)
kde2 = gaussian_kde(long_time)
pdf2 = kde2(x)

from matplotlib.patches import Patch
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots()

# Red:#FE8083   Purple:#8E8EFE   Green:#58D6A6  Gray:#828282   Yellow:#FF9F3A
ax.plot(x, pdf1, '#58D6A6', label='Short', linewidth=1.3)
ax.fill_between(x, pdf1, alpha=0.8, color='#58D6A6')

ax.plot(x, pdf2, '#FE8083', label='Long', linewidth=1.3)
ax.fill_between(x, pdf2, alpha=0.5, color='#FE8083')

ax.plot(x, pdf, '#8E8EFE', label='All', linewidth=1.3)
ax.fill_between(x, pdf,  alpha=0.8, color='#8E8EFE')

ax.set_xlabel('Duration of Flow')
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.set_ylabel('Probability Density')

legend_rect1 = Patch(color="#58D6A6", label='Short')
legend_rect2 = Patch(color="#FE8083", label='Long')
legend_rect3 = Patch(color="#8E8EFE", label='All')


legend = ax.legend(handles=[legend_rect1, legend_rect2, legend_rect3], loc='upper right')
plt.grid(True)
plt.show()
print("Short: {}, Long: {}".format(len(short_time), len(long_time)))
