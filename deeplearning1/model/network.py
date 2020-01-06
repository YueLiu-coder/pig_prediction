import torch.nn as nn
import torch
from model.basic_module import BasicModule
import torch.nn.functional as F
import numpy as np
# 网络结构
class PIGNet(BasicModule):
    def __init__(self):  # 初始化
        super(PIGNet, self).__init__()
        self.model_name = 'PIGNet'
        self.features = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(6, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.full_connected = nn.Sequential(
            nn.Linear(in_features=13 * 1 * 9, out_features=13*5, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=13*5, out_features=1, bias=True),
        )



    def forward(self, x):  # 前向传播，通过前向传播算法输出网络预测值
        ori_x = x
        x = self.features(x)
        x = torch.cat((x, ori_x),1)
        x = x.view(x.size(0), 13 * 1 * 9)
        x = self.full_connected(x)
        return x
    # def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
    #     super(PIGNet, self).__init__()
    #     self.model_name = 'PIGNet'
    #     self.hidden = torch.nn.Linear(in_dim, n_hidden_1)  # 隐藏层线性输出
    #     # self.hidden2 = torch.nn.Linear(n_hidden_1, n_hidden_2)  # 隐藏层线性输出
    #     self.predict = torch.nn.Linear(n_hidden_1, out_dim)  # 输出层线性输出
    #
    #
    # def forward(self, x):
    #     x = self.hidden(x) # 对隐藏层的输出进行relu激活
    #     # x = self.hidden2(x)
    #     x = torch.sigmoid(x)
    #     x = self.predict(x)
    #     return x

    # def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
    #     super(PIGNet, self).__init__()
    #     self.model_name = 'PIGNet'
    #
    #     self.fc = nn.Sequential(
    #     nn.Linear(9, 64),
    #     nn.Sigmoid(),
    #     nn.Linear(64, 20),
    #     nn.Sigmoid(),
    #     nn.Linear(20, 1),)
    #
    # def forward(self, x):
    #     return self.fc(x)
