from model.network import PIGNet
import torch
import torch.nn as nn
import numpy as np
from datasets import listDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import torch.nn.functional as F
from torchnet import meter
import os
from sklearn.metrics import mean_squared_error
import pandas as pd

def test():
    model.eval()  # 模型为测试阶段
    test_data = listDataset('./data/test_datasets2.csv', train=False)  # 读取测试文件

    data = pd.read_csv("./data/test_datasets.csv")
    test_loader = DataLoader(test_data, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers, pin_memory=True)
    result_list = []
    realValues = list(data['饲料量'].values)
    for batch_idx, data in enumerate(test_loader):
        target_pr = model(data)
        result_list += [v[0] for v in target_pr.data.numpy()]


    # MSE

    mse_predict = mean_squared_error(result_list , realValues)
    print(mse_predict)
    pd.DataFrame(result_list).to_excel("./data/test_result.xlsx")


if __name__ == "__main__":
    init_lr = 0.01
    init_epoch = 1
    batch_size = 64
    num_workers = 4
    model = PIGNet()
    pre_train_model = './checkpoints/PIGNet.pth'  # 读取预先训练好的模型
    if pre_train_model:
        model.load(pre_train_model)

    test()
    # data = pd.read_csv('./data/train_datasets2.csv')
    # data = data[data.columns[:-1]]
    # data.to_csv("./data/test_datasets3.csv",index=False)

