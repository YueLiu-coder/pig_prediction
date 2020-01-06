from torch.utils import data
import numpy as np
import pandas as pd
import torch

# 数据导入
class listDataset(data.Dataset):
    def __init__(self, root_dataset, train=True):
        self.train = train
        datasets = pd.read_csv(root_dataset)  # 读取数据
        datasets = datasets.drop(['母猪品种', '体况', '季节'], axis=1)
        datasets = datasets.fillna(-1)  # 填充nan为-1，减少 数据缺失对模型干扰，经测试-1最理想
        self.datasets = np.array(datasets)  # 将dataframe转换为array,此时数据为n×m维，n为样本数量，m为数据维度+label
        if self.train==True:
            pass


    def __len__(self):  # 获取样本数量
        return self.datasets.shape[0]

    def __getitem__(self, index):
        if self.train==True:  # 训练情况下，即有样本标签

            example = torch.from_numpy(self.datasets[index][:-1].reshape(1, -1)).float()
            label = torch.from_numpy(self.datasets[index][-1].reshape(-1)).float()

            return example, label
        else:  # 测试情况下，即没有样本标签
            example = torch.from_numpy(self.datasets[index].reshape(1, -1)).float()
            return example
