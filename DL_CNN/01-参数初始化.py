import torch
import matplotlib.pyplot as plt

# 均匀分布，随机初始化
def demo01():

    # 定义模型，一层全连接层
    linear = torch.nn.Linear(128, 256)


    # 对权重进行初始化
    # 参1：待初始化的参数 参2：权重参数的初始化范围
    torch.nn.init.uniform_(linear.weight, a=-1, b=1)
    torch.nn.init.uniform_(linear.bias, a=-1, b=1) # 对偏置进行初始化

    print(linear.weight.data,linear.bias.data)
    print(linear.weight.data.shape,linear.bias.data.shape)


    # 绘制柱状图，观察权重和偏置的分布
    plt.hist(linear.weight.data.numpy().flatten(), 50)
    plt.show()

# 正太分布，随机初始化
def demo02():
    # 定义模型，一层全连接层
    linear = torch.nn.Linear(128, 256)

    # 对权重进行初始化
    # 参数1：待初始化的参数 参2：均值 参3：标准差
    torch.nn.init.normal_(linear.weight, mean=0, std=1)
    torch.nn.init.normal_(linear.bias, mean=0, std=1)

    print(linear.weight.data,linear.bias.data)
    print(linear.weight.data.shape,linear.bias.data.shape)

    # 绘制柱状图，观察权重和偏置的分布
    # 参数1：待绘制的数组,将张量转为numpy后用flatten()展平  参2：柱状图宽度
    plt.hist(linear.weight.data.numpy().flatten(), 50)
    plt.show()

# 固定值随机初始化
def demo03():
    # 创建一个全连接层
    linear = torch.nn.Linear(128, 256)

    # 对权重进行初始化
    # torch.nn.init.constant_(linear.weight, val=0.1)
    # 全零初始化
    torch.nn.init.zeros_(linear.weight)

    print(linear.weight.data)
    print(linear.weight.data.shape)

    # 绘制柱状图，观察权重和偏置的分布
    plt.hist(linear.weight.data.numpy().flatten(), 50)
    plt.show()


# kaiming 初始化
def demo04():
    # 创建一个全连接层
    linear = torch.nn.Linear(128, 256)

    # 对权重进行初始化
    # torch.nn.init.kaiming_normal_(linear.weight) # 标准差 0.125
    torch.nn.init.kaiming_uniform_(linear.weight) # 范围： [-sqrt(k), sqrt(k)] -0.216 ~ 0.216


    print(linear.weight.data)
    print(linear.weight.data.shape)


    plt.hist(linear.weight.data.numpy().flatten(), 50)
    plt.show()


# xavier 初始化
def demo05():
    # 创建一个全连接层
    linear = torch.nn.Linear(128, 256)


    # torch.nn.init.xavier_normal_(linear.weight) # std = sqrt(2 / (fan_in + fan_out)) = sqrt(2 / (128 + 256)) = 0.072
    torch.nn.init.xavier_uniform_(linear.weight) # limit = sqrt(6 / (fan_in + fan_out)) = 0.125  范围：-0.125 ~ 0.125

    print(linear.weight.data)
    print(linear.weight.data.shape)

    plt.hist(linear.weight.data.numpy().flatten(), 50)
    plt.show()



















if __name__ == '__main__':
    # demo01()
    # demo02()
    # demo03()
    # demo04()
    demo05()