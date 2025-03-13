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

class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(DepthwiseSeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

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

        self.conv3d_2 = DepthwiseSeparableConv3D(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.conv3d_bn2 = nn.BatchNorm3d(64)

        # 2D卷积层
        self.conv2d_1 = DepthwiseSeparableConv2D(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2d_bn1 = nn.BatchNorm2d(128)

        self.conv2d_2 = DepthwiseSeparableConv2D(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2d_bn2 = nn.BatchNorm2d(256)

        self.conv2d_3 = DepthwiseSeparableConv2D(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2d_bn3 = nn.BatchNorm2d(512)

        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

        # LSTM层
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # print("Input shape:", x.shape)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 4:
            x = x.unsqueeze(1)
        assert x.dim() == 5, "Input to conv3d must be a 5D tensor"

        # print("After unsqueeze, shape:", x.shape)
        # 3D卷积层
        x = F.relu(self.conv3d_1(x))
        x = self.conv3d_bn1(x)

        # print("After first conv3d, shape:", x.shape)

        x = F.relu(self.conv3d_2(x))
        x = self.conv3d_bn2(x)

        # print("After second conv3d, shape:", x.shape)

        # 调整维度顺序并展平
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size, channels, height * depth, width)
        x = x.permute(0, 1, 3, 2).contiguous()

        # print("After reshape and permute, shape:", x.shape)
        # 2D卷积层
        x = F.relu(self.conv2d_1(x))
        x = self.conv2d_bn1(x)

        x = F.relu(self.conv2d_2(x))
        x = self.conv2d_bn2(x)

        x = F.relu(self.conv2d_3(x))
        x = self.conv2d_bn3(x)

        # print("After conv2d, shape:", x.shape)
        # Dropout
        x = self.dropout(x)

        # 将2D卷积层输出转换为LSTM输入
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height, width * channels)

        # print("Before LSTM, shape:", x.shape)
        # 确保LSTM输入大小正确
        lstm_input_size = width * channels
        if lstm_input_size != 512:
            print(f"Warning: LSTM input size mismatch. Expected 512, but got {lstm_input_size}")

        # LSTM层
        x, _ = self.lstm(x)

        # print("After LSTM, shape:", x.shape)

        x = x[:, -1, :]

        # print("After LSTM (last time step), shape:", x.shape)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print("After fully connected layers, shape:", x.shape)
        return x