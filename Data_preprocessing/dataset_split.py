from sklearn.model_selection import train_test_split

import numpy as np

def random_dropout(data, dropout_rate):
    """
    随机舍弃一部分数据集。

    参数：
    - data: 要进行舍弃的数据集，可以是列表、数组或其他可迭代对象。
    - dropout_rate: 舍弃的比例，取值范围为 [0, 1]。

    返回：
    - 舍弃后的数据集。
    """
    if not 0 <= dropout_rate <= 1:
        raise ValueError("dropout_rate 应在 [0, 1] 范围内")

    # 计算要舍弃的样本数量
    num_samples_to_drop = int(len(data) * dropout_rate)

    # 随机选择要舍弃的样本索引
    indices_to_drop = np.random.choice(len(data), num_samples_to_drop, replace=False)

    # 舍弃选定的样本
    remaining_data = [sample for i, sample in enumerate(data) if i not in indices_to_drop]

    return remaining_data


# 读取txt文件
protocl = "multi_4cls"
filepath = './protocols/'
with open(filepath + protocl+'.txt', 'r', encoding='utf-8') as file:
# with open('cip_without_coding.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 假设每一行是一个样本，可以根据实际情况进行处理
data = [line.strip() for line in lines]

# 划分训练集和验证集
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 将训练集写入新的txt文件
with open(filepath + protocl+'_train_data.txt', 'w', encoding='utf-8') as file:
    for sample in train_data:
        file.write(sample + '\n')

# 将验证集写入新的txt文件
with open(filepath + protocl+'_val_data.txt', 'w', encoding='utf-8') as file:
    for sample in val_data:
        file.write(sample + '\n')

"""  Datasets scale split
scale = [2, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25]
for i in range(len(scale)):
    dropout_rate = 1-(scale[i]/scale[0])
    remaining_data = random_dropout(data, dropout_rate)
    print("---------------/ {} /----------------".format(scale[i]))
    print("Original Data:", len(data))
    print("Remaining Data:", len(remaining_data))

    # 划分训练集和验证集
    train_data, val_data = train_test_split(remaining_data, test_size=0.2, random_state=42)

    # 将训练集写入新的txt文件
    with open('scale_set/'+str(scale[i])+'_train_data.txt', 'w', encoding='utf-8') as file:
        for sample in train_data:
            file.write(sample + '\n')

    # 将验证集写入新的txt文件
    with open('scale_set/'+str(scale[i])+'_val_data.txt', 'w', encoding='utf-8') as file:
        for sample in val_data:
            file.write(sample + '\n')
"""