import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, dropout, num_classes=7, max_len=400):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12736, max_len)
        self.fc2 = nn.Linear(max_len, num_classes)  # 输出层，根据你的类别数量确定输出维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TimeSeriesCNNFCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TimeSeriesCNNFCN, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # FCN layers
        self.fcn = nn.Sequential(
            nn.Linear(128 * (input_size // 4), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Reshape input for 1D convolution
        x = x.view(x.size(0), 1, -1)

        # CNN layers
        x = self.cnn(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # FCN layers
        x = self.fcn(x)

        return x

# 示例用法
# input_size = 100  # 替换为你的时间序列长度
# num_classes = 10  # 替换为你的类别数量
#
# model = TimeSeriesCNNFCN(input_size, num_classes)
# print(model)
