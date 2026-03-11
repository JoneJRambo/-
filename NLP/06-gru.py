import torch
import torch.nn as nn


# 一个样本, 一个 token
def dm01_gru():
    # 构建模型
    gru = nn.GRU(input_size=4, hidden_size=6, num_layers=1)
    # 构建数据
    # 参1: sequence_length,句子长度,也就是这句话中的token数量
    # 参2: batch_size,批次大小,也就是样本数量
    # 参3: input_size,输入的数据维度,也就是词向量的维度
    input = torch.randn(1, 1, 4)
    # 参1: num_layers, 隐藏层层数,默认为1
    # 参2: batch_size,批次大小,也就是样本数量
    # 参3: hidden_size,隐藏层维度,也就是隐藏层的特征数
    h0 = torch.zeros(1, 1, 6)
    # 把数据送给模型
    output, hn = gru(input, h0)
    # 输出结果
    print('output-->', output)
    print('hn-->', hn)



# 多个样本,多个token (3个样本,4个token,GRU模型,输出维度是4,输出是6)
def dm02_gru():
    gru = nn.GRU(input_size=4, hidden_size=6, num_layers=1)
    input = torch.randn(4,3,4)
    h0 = torch.randn(1,3,6)
    output, hn = gru(input, h0)
    print('output-->', output)
    print('hn-->', hn)


if __name__ == '__main__':
    # dm01_gru()
    dm02_gru()