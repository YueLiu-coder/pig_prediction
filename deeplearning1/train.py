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
import random

def adjust_learning_rate(optimizer, epoch):  # 调整学习率
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    if epoch % 200 == 0:
        lr = init_lr * (0.1 ** (epoch // 200))
        print('current lr: ', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# 训练的主要内容
def train(epoch):
    model.train()  # 将模型变成训练模式
    train_data = listDataset('./data/train_datasets.csv', train=True)  # 获取数据，这里的地址填写生成样本的地址
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_data = listDataset('./data/test_datasets.csv', train=True)  # 获取数据，这里的地址填写生成样本的地址
    test_loader = DataLoader(test_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    loss_meter = meter.AverageValueMeter()  # loss记录参数
    loss_meter.reset()
    test_data_list = [(index,test_data_v,test_data_t) for index,(test_data_v,test_data_t) in enumerate(test_loader)]
    for batch_idx, (data, target) in enumerate(train_loader):  # 进入训练
        # target = target.
        # data, target = data.cuda(), target.cuda()
        index,test_data_temp,test_data_target = test_data_list[0]
        # print(test_data_temp)
        optimizer.zero_grad()  # 每次inference前将梯度清零
        target_pr = model(data)  # 模型通过输入数据输出预测饲料量？？？训练集？
        target_pr2 = model(test_data_temp)  # 模型通过输入数据输出预测饲料量？？？验证集？
        # target_pr = target_pr.cuda()
        loss = F.mse_loss(target.float(), target_pr)  # loss计算，这里采用MSE方式
        loss2 = F.mse_loss(test_data_target.float(), target_pr2)
        loss.backward()  # 梯度反向传播？更新网络参数
        optimizer.step()  # 优化器优化阶段
        loss_meter.add(loss.item())  # 将loss记录

        if batch_idx % 10 == 0:  # 预览loss情况？？？10是什么意思？

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t''Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            with open("./loss1.txt","a+") as fp:
                fp.write("%s\n"%loss.item())


            with open("./loss2.txt","a+") as fp:
                fp.write("%s\n"%loss2.item())


    if epoch % 200 == 0:  # 每200epoch保存一次样本
        model.save()



if __name__ == "__main__":
    init_lr = 0.001  # 设置初始学习率
    init_epoch = 1  # 设置初始epoch
    max_epoch = 601  # 设置最大epoch
    batch_size = 64  # 设置每次训练的样本数
    num_workers = 4  # 超线程读取数据
    # in_dim, n_hidden_1, n_hidden_2, out_dim = 9,128,64,1
    model = PIGNet()
    pre_train_model = ''
    if pre_train_model:
        model.load(pre_train_model)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    for epoch in range(init_epoch, max_epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
