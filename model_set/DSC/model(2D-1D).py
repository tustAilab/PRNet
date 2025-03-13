import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ProteinRankNet(nn.Module):
    def __init__(self, input_shape):
        super(ProteinRankNet, self).__init__()

        # 3D卷积层
        self.conv3d_1 = DepthwiseSeparableConv3D(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.conv3d_bn1 = nn.BatchNorm3d(32)

        # 添加更多的3D卷积层
        self.conv3d_2 = DepthwiseSeparableConv3D(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.conv3d_bn2 = nn.BatchNorm3d(64)

        # 1D卷积层
        self.conv1d_1 = DepthwiseSeparableConv1D(64 * 20, 128, kernel_size=3, stride=2, padding=1)
        self.conv1d_bn1 = nn.BatchNorm1d(128)

        self.conv1d_2 = DepthwiseSeparableConv1D(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv1d_bn2 = nn.BatchNorm1d(256)
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

        # LSTM层
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 2:  # 检查维度，如果是2D张量，调整为5D
            x = x.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        elif x.dim() == 3:  # 检查维度，如果是3D张量，调整为5D
            x = x.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 4:  # 检查维度，如果是4D张量，调整为5D
            x = x.unsqueeze(1)
        assert x.dim() == 5, "Input to conv3d must be a 5D tensor"

        # 3D卷积和池化层
        x = F.relu(self.conv3d_1(x))
        x = self.conv3d_bn1(x)
        
        x = F.relu(self.conv3d_2(x))
        x = self.conv3d_bn2(x)

        # 调整维度顺序并展平
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size, channels * depth, height * width)  # 变为 (batch_size, channels * depth, new_width)

        # 1D卷积层
        x = F.relu(self.conv1d_1(x))
        x = self.conv1d_bn1(x)

        x = F.relu(self.conv1d_2(x))
        x = self.conv1d_bn2(x)

        # Dropout
        x = self.dropout(x)

        # 将1D卷积层输出转换为LSTM输入
        x = x.permute(0, 2, 1).contiguous()  # 变为 (batch_size, seq_len, features)

        # LSTM层
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x