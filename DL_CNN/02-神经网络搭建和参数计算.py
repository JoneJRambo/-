import torch
import torch.nn as nn
from torchsummary import summary

"""
    神经网络搭建流程：
        1.定义一个类，继承：nn.Module
        2.在__init__()方法中，搭建神经网络
        3.在forward()方法中，完成：前向传播
"""

class Net(nn.Module):
    def __init__(self):
        # 初始化父类成员
        super(Net, self).__init__()

        # 定义第一个隐藏层 3进3出
        self.linear1 = nn.Linear(3,3)

        # 定义第二个隐藏层 3进2出，当前层输入维度等于前一层输出维度
        self.linear2 = nn.Linear(3,2)

        # 输出层 2进2出
        self.linear3 = nn.Linear(2,2)

        # 对两个隐藏层进行初始化
        # 第一层用Xavier初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 第二层用Kaiming初始化
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        #第一层用sigmoid,第二层用ReLU
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()

    def forward(self,input):
        # 将input载入到第一个隐藏层，并激活
        out1 = self.activation1(self.linear1(input))
        # out1 = torch.sigmoid(self.linear1(input))  # 等价形式
        # o1 = self.linear1(input)
        # out1 = self.activation1(o1)

        # 将out1载入第二个隐藏层，并激活
        out1 = self.activation2(self.linear2(out1)) # 也用out1命名，降低内存占用

        # 将第二层输出载入输出层
        out1 = self.linear3(out1)
        out1 = torch.softmax(out1, dim=1)

        return out1


if __name__ == '__main__':
    # 创建神经网络
    net = Net()

    # 一行代码：自动检测并移到可用设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # 创建输入数据并移到相同设备
    result = torch.randn(5, 3).to(device)
    # input = torch.randn(5, 3)

    # 获取输出
    output = net(result)
    print(output)
    print(output.shape)
    # 获取网络结构
    summary(net, (5,3), device='cuda')












