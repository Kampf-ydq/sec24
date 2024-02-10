import numpy as np
import matplotlib.pyplot as plt

flow_lgths = [6, 890, 3, 216, 328, 160, 526, 86, 1904, 12, 394, 548, 110, 1, 1345, 2, 344, 4, 64, 1174, 4, 834, 1006, 10, 20,
 292, 228, 370, 70, 16, 20, 10, 20, 194, 376, 70, 152, 94, 3190, 530, 228, 264, 150, 36, 150, 74, 12, 120, 28, 68,
 106, 506, 14, 268, 14, 66, 482, 48, 8, 184, 34, 84, 16, 28, 38, 74, 32, 68, 30, 8, 10, 30, 46, 8, 48, 20, 4, 18,
 304, 116, 738, 302, 284, 88, 436, 52, 114, 208, 40, 78, 940, 72, 62, 522, 400, 2, 136, 74, 8, 2, 100, 2, 10, 42,
 46, 66, 20, 66, 22, 18, 204, 164, 178, 152, 6, 34, 238, 158, 252, 934, 552, 154, 68, 34, 34, 8, 246, 120, 660, 400,
 14, 542, 236, 20, 114, 34, 2, 26, 18, 38, 10, 98, 40, 26, 80, 94, 178, 80, 68, 142, 320, 166, 148, 390, 686, 270,
 94, 576, 116, 28, 10, 6, 6, 8, 192, 62, 592, 6, 10, 122, 778, 196, 88, 558, 252, 350, 666, 644, 118, 498, 178, 6,
 4, 34, 4, 86, 340, 186, 392, 1972, 172, 334, 86, 210, 340, 114, 164, 68, 354, 52, 6, 12, 16, 294, 64, 38, 4, 4, 6,
 160, 252, 154, 542, 186, 728, 250, 164, 258, 402]
# Calculation of cumulative percentage
x_sorted = np.sort(flow_lgths)[::-1]
cumulative_percentage = np.cumsum(x_sorted) / np.sum(x_sorted) * 100


FLOW_SIZE = 300
indices = np.where(x_sorted > FLOW_SIZE)
indices = indices[0][-1]

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots()


# Scatter size
long_scatter = 2000/cumulative_percentage[:indices]
short_scatter = 1000/cumulative_percentage[indices:]

ax.scatter([i for i in range(indices)], cumulative_percentage[:indices], s=long_scatter, color='#FE8083', label='Long')
ax.scatter([i for i in range(indices,len(cumulative_percentage))], cumulative_percentage[indices:], s=short_scatter, color='#58D6A6', label='Short')

ax.set_xlabel('Flow Number')
ax.set_ylabel('Cumulative Percentage of Messages')
plt.legend()

plt.grid(True)
plt.show()

