'''
案例：
利用手机价格区间的预测的案例，巩固深度学习模型搭建以及训练过程的知识

步骤：
1 准备训练集数据
2 构建要使用的模型
3 模型训练
4 模型预测评估
'''

# 导包
# 导入相关模块
import torch  # pytorh
from torchsummary import summary
from torch.utils.data import TensorDataset  # 创建数据集对象,定义了数据集的加载方式，以及数据增强的算法。
from torch.utils.data import DataLoader  # 数据分发器（加载器）
import torch.nn as nn  # pytorch下属的神经网络的封装
import torch.optim as optim  # 优化器模块
from sklearn.model_selection import train_test_split  # 训练集和测试集的分割
import numpy as np  # 处理数组
import pandas as pd  # 表格数据处理


#1 准备训练集数据
def create_dataset():
    path ='datasets\手机价格预测.csv'
    # 用pd 读取表格数据
    data = pd.read_csv(path)
    #print(data.head(), data.shape)

    # 获得数据集的 特征列 以及 标签列
    x = data.iloc[:, :-1] ;  y = data.iloc[:, -1]
    #print(x.shape,y.shape)    # (2000, 20)    (2000)

    #将数据集的数据类型转换为浮点类型    标签应该是整型
    x = x.astype(np.float32) ; y = y.astype(np.int64)

    # 将数据集进行训练集和测试集的分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print(len(x_train),type(x_train),x_train )  #  2000个数据被划分为 1600 ： 400条数据

    # 2244    2 ： 数据集对象  --》 dataloader
    #  将numpy数组转换为张量
    x_train = torch.from_numpy(x_train.values.copy())
    x_test = torch.from_numpy(x_test.values.copy())
    y_train = torch.from_numpy(y_train.values.copy())
    y_test = torch.from_numpy(y_test.values.copy())


    # 进行数据集的封装
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_test, y_test)

    #print(len(train_dataset), type(train_dataset))

    # 展示标签，以及标签的种类数
    print(y_train.unique(), len(y_train.unique()))
    class_number = len(y_train.unique())

    # 返回的内容： 训练集对象，测试集对象，特征维度，标签种类数
    return train_dataset, valid_dataset, x_train.shape[1] ,class_number



# 搭建神经网络模型
class phone_price_model(nn.Module):

    # 定义模型结构 魔法方法
    # 需要传入特征维度，标签种类数
    def __init__(self, feature_number, class_number):
        super().__init__()

        # 定义隐藏层第一层 输入是20个输入特征，输出是128个特征（该层提取了128份高阶特征，该层由128个神经元）
        self.fc1 = nn.Linear(feature_number, 128)  # 输入层到隐藏层的全连接层

        # 隐层第二层，输入是128个特征（前一层的输出维度），输出是256个特征
        self.fc2 = nn.Linear(128, 256)  # 隐藏层到隐藏层的全连接层

        # 定义输出层，也是全连接层，输入维度是256，输出维度是标签种类数
        self.fc3 = nn.Linear(256, class_number)  # 隐藏层到输出层的全连接层
        # 定义激活层
        self.act = nn.ReLU()  # 激活函数

    # 定义前向传播（数据流）， 定义输入数据的形式
    def forward(self, x):

        #x = self.fc1(x)    # 将输入数据送入第一层全连接层，得到输出
        #x = self.act(x)    #  将输出数据送入激活层，得到输出
        x = self.act(self.fc1(x))  # 等价形式，嵌套写法

        x = self.act(self.fc2(x))  # 嵌套写法，第一层的输出作为第二层的输入，得到结果

        x = self.fc3(x)  # 输出层，得到结果 （每个手机样本，在4个价格区间的分类得分）

        return x  # 返回模型的预测结果



# 2244 口诀
# 2 创建数据集对象   数据加载器 dataloader
# 2  2个循环嵌套    第一层循环是对(epoch) 数据集进行迭代 每一轮迭代，等于将训练集送入模型一轮
#                  第二层循环，是在每一轮迭代中，将数据集进行分批次送入模型进行训练 ，即按批次进行迭代
# 4 定义模型 定义损失函数 优化器  学习率
# 4 在训练过程中，进行梯度清零  反向传播求梯度  参数更新。  学习率的更新判断

# 撰写训练脚本
def train_model(train_dataset,feature_number, class_number, device):

    # 定义数据加载器 （定义了每个训练批次 随机抽取多少数据 喂入模型）
    # 参数1 ： 数据集对象  参数2 ： 批次大小  参数3： 是否打乱数据  参数4： 是否丢弃最后不满批次的数据
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

    # 定义模型  模型实例化
    model = phone_price_model(feature_number, class_number).to(device)

    # 定义损失函数  多分类
    criterion = nn.CrossEntropyLoss()

    # 定义优化器  随机梯度下降
    # 参数1 指定模型参数  参数2 ： 学习率（恒定）
    optimizer = optim.SGD(model.parameters(), lr=0.001)


    # 2个循环
    for epoch in range(500):  # 对 轮数进行循环

        loss_sum = 0  # 定义 批次累计损失
        iter_num =0  # 定义训练的批次数


        # 在一个轮次中，对批次进行循环  dataloader在每次迭代，都会返回一个当前批次 随机抽中的数据，来训练模型
        # 这数据的形式 为 (批次数据，批次标签)
        for step, (x_batch, y_true) in enumerate(dataloader):
            # 将当前批次的数据，送入gpu
            x_batch = x_batch.to(device)
            y_true = y_true.to(device)

            # 将当前批次的数据，送入模型进行训练
            y_pred = model(x_batch)

            # 计算交叉熵损失  # 真实值需要整型  预测值是浮点
            loss = criterion(y_pred, y_true)

            #  梯度清零
            optimizer.zero_grad()

            # 反向传播求梯度
            loss.backward()

            # 参数更新
            optimizer.step()

            #做记录，记录训练过程中，每个批次的loss，以及训练了几个批次
            loss_sum += loss.item()
            iter_num += 1

        # 计算当前轮训练的平均loss
        loss_mean = loss_sum / iter_num
        print('epoch=',epoch,'  loss=',loss_mean)

    # 训练完毕，保存模型参数
    # 参1 ，定义模型参数  参数2 ： 模型参数保存的路径
    torch.save(model.state_dict(), 'phone_price_model.pth')


# 定义模型推理函数
def eval_model(valid_dataset, feature_number, class_number, device):
    # 定义数据加载器（定义了每个训练批次 随机抽取多少数据 喂入模型）
    # 参数1： 数据集对象 参数2： 批次大小 参数3： 是否打乱数据
    dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    # 定义模型  模型实例化
    model = phone_price_model(feature_number, class_number).to(device)

    # 加载模型参数
    # 1. 读取训练好的模型参数 （文件pth)
    # 2. 将读入的模型参数，加载到模型结构中
    model.load_state_dict(torch.load('phone_price_model.pth', map_location=device))

    model.eval()

    # 初始化准确度得分
    acc_score = 0

    # 定义一个循环，对每个测试样本进行推理
    for step, (x_batch, y_true) in enumerate(dataloader):
        # print(x_batch,x_batch.shape)
        # print(y_true)
        # 关键修改：将数据移到GPU
        x_batch = x_batch.to(device)
        y_true = y_true.to(device)

        # 正向传播
        y_pred = model(x_batch)
        # print(y_pred)

        # 获取最高分类得分的下标
        y_pred_index = torch.argmax(y_pred, dim=1)
        # print(y_pred_index)

        # 当我的预测值和真实值一致时，准确度加1
        if y_pred_index.item() == y_true.item():
            acc_score += 1

    # 输出模型准确度
    print('acc_score=',acc_score/len(valid_dataset))




if __name__ == '__main__':

    device = torch.device('cuda')

    train_dataset, valid_dataset, feature_number, class_number= create_dataset()

    # 搭建神经网络的模型
    model = phone_price_model(feature_number, class_number).to(device)

    # 打印模型结构

    # 训练过程
    train_model(train_dataset, feature_number, class_number, device)

    # 模型推理
    eval_model(valid_dataset, feature_number, class_number, device)