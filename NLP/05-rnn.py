import torch
import torch.nn as nn


# 1个样本,该样本只有一个特征
def dm01():
    # 准备数据
    # 参1: sequence_length,句子长度,也就是这句话中的token数量
    # 参2: batch_size,批次大小,也就是样本数量
    # 参3: input_size,输入的数据维度,也就是词向量的维度
    input = torch.randn(1, 1, 4)
    # 参1: num_layers, 隐藏层层数,默认为1
    # 参2: batch_size,批次大小,也就是样本数量
    # 参3: hidden_size,隐藏层维度,也就是隐藏层的特征数
    h0 = torch.zeros(1, 1, 6)
    # 构建RNN模型
    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=1)
    # 把数据送给模型
    output, hn = rnn(input, h0)
    # 输出结果
    print('output-->', output)
    print('hn-->', hn)

    # 结论:
    # 1. 深度神经网络模型,只能对特征维度进行缩放,不能改变批次叔和token数,因此[1,1,a] ->[1,1,b]
    # 2. h0和hn的shape是一样的


# 1个样本,多个token
def dm02():
    # 构建模型
    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=1)
    # 准备数据
    # (sequence_len, batch_size, input_size)
    input = torch.randn(2, 1, 4)
    # (num_layers, batch_size, hidden_size)
    h0 = torch.randn(1, 1, 6)
    # 把数据送给模型
    output, hn = rnn(input, h0)
    # 输出
    print('output-->', output)  # [2, 1, 6]
    print('hn-->', hn)  # [1, 1, 6]


# 多条样本,多个token
def dm03():
    # 构建模型
    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=1)
    # 构建数据
    input = torch.randn(5, 3, 4)
    h0 = torch.randn(1, 3, 6)
    output, hn = rnn(input, h0)
    print('output-->', output)
    print('hn-->', hn)


# 多条样本,多个token,多个隐藏层
def dm04():
    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=2)
    input = torch.randn(5, 3, 4)
    h0 = torch.randn(2, 3, 6)
    output, hn = rnn(input, h0)
    print('output-->', output)
    print('hn-->', hn)


# 省略h0参数
def dm05():
    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=2)
    input = torch.randn(5, 3, 4)
    h0 = torch.randn(2, 3, 6)
    output, hn = rnn(input)
    #  省略h0参数
    print('output-->', output)
    print('hn-->', hn)
    # 不省略h0参数
    output, hn = rnn(input, h0)
    print('output-->', output)
    print('hn-->', hn)


def dm06():.
    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=2)
    input = torch.randn(3, 5, 4)
    h0 = torch.randn(2, 5, 6)

    # 一个样本一个样本送给模型
    for i in range(input.shape[1]):
        x0 = input.transpose(0, 1)[i].unsqueeze(dim=1)
        h0 = torch.zeros(2, 1, 6)
        output, hn = rnn(x0, h0)
        print(f'output{i}-->', output)


# batch_first = True
def dm07():
    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=2, batch_first=True)
    input = torch.randn(5, 3, 4)
    input = input.transpose(0, 1)
    # input = torch.randn(3,5,4)
    h0 = torch.randn(2, 3, 6)
    output, hn = rnn(input, h0)
    print('output-->', output)


if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm05()
    # dm06()
    dm07()
