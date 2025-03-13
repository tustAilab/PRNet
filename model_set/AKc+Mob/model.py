import torch
import torch.nn as nn
import torch.nn.functional as F
from AKConv import AKConv

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ProteinRankNet(nn.Module):
    def __init__(self, input_shape):
        super(ProteinRankNet, self).__init__()

        # 3D卷积层
        self.conv3d_1 = DepthwiseSeparableConv3D(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_bn1 = nn.BatchNorm3d(16)
        
        # 添加更多的3D卷积层
        self.conv3d_2 = DepthwiseSeparableConv3D(16, 32)
        self.conv3d_bn2 = nn.BatchNorm3d(32)
        

        # AKConv 2D卷积层
        self.akconv2d_1 = AKConv(32, 64, num_param=3)
        self.akconv2d_bn1 = nn.BatchNorm2d(64)
        
        # 添加更多的AKConv 2D卷积层
        self.akconv2d_2 = AKConv(64, 128, num_param=3)
        self.akconv2d_bn2 = nn.BatchNorm2d(128)
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

        # LSTM层
        self.lstm = nn.LSTM(input_size=128 * 20, hidden_size=64, num_layers=2, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        if x.dim() == 3:  # 检查维度
            x = x.unsqueeze(1)  # 添加一个维度，变为 (batch_size, 1, depth, height, width)
            x = x.unsqueeze(2)  # 再添加一个维度，变为 (batch_size, 1, 1, depth, height, width)
        assert x.dim() == 5, "Input to conv3d must be a 5D tensor"

        # 3D卷积和池化层
        x = F.relu(self.conv3d_1(x))
        x = self.conv3d_bn1(x)
        x = F.relu(self.conv3d_2(x))
        x = self.conv3d_bn2(x)

        # 调整维度顺序并展平
        x = x.view(x.size(0), x.size(1), -1, x.size(-1))  # 变为 (batch_size, channels, depth * height, width)
        x = x.permute(0, 1, 3, 2).contiguous()  # 变为 (batch_size, channels, width, depth * height)

        # AKConv 2D卷积层
        x = F.relu(self.akconv2d_1(x))
        x = self.akconv2d_bn1(x)
        x = F.relu(self.akconv2d_2(x))
        x = self.akconv2d_bn2(x)

        # Dropout
        x = self.dropout(x)

        # 将2D卷积层输出转换为LSTM输入
        x = x.view(x.size(0), -1, 128 * 20)  # 调整为 (batch_size, seq_len, features)

        # LSTM层
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x