import torch
from torch import nn
import torch.nn.functional as F

# ==========================================
# 1. CIFAR-10 网络架构 (3通道，10分类默认)
# ==========================================
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# ==========================================
# 2. FEMNIST 网络架构 (1通道，62分类默认)
# ==========================================
class CNNEmnist(nn.Module):
    def __init__(self, args):
        super(CNNEmnist, self).__init__()
        # FEMNIST 是单通道灰度图，输入通道为 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 28x28 经过两次 conv 和一次 max_pool 后，特征图大小对应 9216
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # 展平操作 (依赖 import torch)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 3. MLP_Loan (返回 Raw Logits)
# ==========================================
class MLP_Loan(nn.Module):
    def __init__(self, dim_in, dim_hidden=64, dim_out=2):
        super(MLP_Loan, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        # 🚨 拯救 Loan 的核心护盾
        self.ln1 = nn.LayerNorm(dim_hidden)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.layer_hidden = nn.Linear(dim_hidden, dim_hidden)
        self.ln2 = nn.LayerNorm(dim_hidden)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.layer_out = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer_hidden(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.layer_out(x)
        return x