import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAM

class ProteinRankNet(nn.Module):
    def __init__(self, input_shape):
        super(ProteinRankNet, self).__init__()

        # 3D卷积层
        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_bn1 = nn.BatchNorm3d(32)
        
        # 添加更多的3D卷积层
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_bn2 = nn.BatchNorm3d(64)
        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_bn3 = nn.BatchNorm3d(128)

        # 添加CBAM模块
        self.cbam1 = CBAM(128)

        # 2D卷积层
        self.conv2d_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv2d_bn1 = nn.BatchNorm2d(256)
        
        # 添加更多的2D卷积层
        self.conv2d_2 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.conv2d_bn2 = nn.BatchNorm2d(512)

        # 添加第二个CBAM模块
        self.cbam2 = CBAM(512)
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

        # LSTM层
        self.lstm = nn.LSTM(input_size=512 * 20, hidden_size=256, num_layers=2, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

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
        x = F.relu(self.conv3d_3(x))
        x = self.conv3d_bn3(x)

        # 调整维度以应用CBAM模块1
        x = x.view(x.size(0), -1, x.size(3), x.size(4))  # 变为 (batch_size, channels * depth, height, width)
        x = self.cbam1(x)

        # 调整维度以恢复3D卷积格式
        depth = 128  # 这里的depth是128，因为它是之前3D卷积输出的深度维度
        x = x.view(x.size(0), depth, x.size(2), x.size(3))  # 变为 (batch_size, channels, height, width)

        # 调整维度顺序并展平
        x = x.view(x.size(0), x.size(1), -1, x.size(-1))  # 变为 (batch_size, channels, depth * height, width)
        x = x.permute(0, 1, 3, 2).contiguous()  # 变为 (batch_size, channels, width, depth * height)

        # 2D卷积层
        x = F.relu(self.conv2d_1(x))
        x = self.conv2d_bn1(x)
        x = F.relu(self.conv2d_2(x))
        x = self.conv2d_bn2(x)

        # CBAM模块2
        x = self.cbam2(x)

        # Dropout
        x = self.dropout(x)

        # 将2D卷积层输出转换为LSTM输入
        x = x.view(x.size(0), -1, 512 * 20)  # 调整为 (batch_size, seq_len, features)

        # LSTM层
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x