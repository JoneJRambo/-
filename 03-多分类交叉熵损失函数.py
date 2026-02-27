import torch
import torch.nn as nn

# 定义函数，演示多分类交叉熵损失函数
def demo01():

    # 手动创建样本的真实标签
    y_true = torch.tensor([[0,1,0],[0,0,1]], dtype=torch.float)

    # 创建分类网络的输出结果（3个类别上的得分） 共2个样本，3个类别
    y_pred = torch.tensor([[-0.02,0.6,2.36],[0.6,0.8,-0.3]],requires_grad= True, dtype=torch.float)

    # 创建多分类交叉熵损失函数对象，其内部已包含 softmax 激活函数
    criterion = nn.CrossEntropyLoss()

    # 设置温度系数 0.8
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100, label_smoothing=0.8)

    # 计算损失函数值
    loss = criterion(y_pred, y_true)
    # 将loss 转换为numpy数组
    loss = loss.detach().numpy()
    print(loss)


if __name__ == '__main__':
    demo01()