from sklearn.model_selection import train_test_split

import numpy as np

def random_dropout(data, dropout_rate):
    """
    Randomly discard part of the data set.
    """
    if not 0 <= dropout_rate <= 1:
        raise ValueError("dropout_rate must be [0, 1].")

    num_samples_to_drop = int(len(data) * dropout_rate)
    indices_to_drop = np.random.choice(len(data), num_samples_to_drop, replace=False)

    remaining_data = [sample for i, sample in enumerate(data) if i not in indices_to_drop]
    return remaining_data


protocl = "multi_4cls"
filepath = './protocols/'
with open(filepath + protocl+'.txt', 'r', encoding='utf-8') as file:
# with open('cip_without_coding.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

data = [line.strip() for line in lines]
train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

# Write the training set to a new txt file
with open(filepath + protocl+'_train_data.txt', 'w', encoding='utf-8') as file:
    for sample in train_data:
        file.write(sample + '\n')

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

    train_data, val_data = train_test_split(remaining_data, test_size=0.2, random_state=42)

    with open('scale_set/'+str(scale[i])+'_train_data.txt', 'w', encoding='utf-8') as file:
        for sample in train_data:
            file.write(sample + '\n')

    with open('scale_set/'+str(scale[i])+'_val_data.txt', 'w', encoding='utf-8') as file:
        for sample in val_data:
            file.write(sample + '\n')
"""