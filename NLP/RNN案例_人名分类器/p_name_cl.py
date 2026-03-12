import string
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# TODO 1.数据预处理
# TODO: 在工作中,TODO代表未完待续,在学习中,高亮显示
# 准备字母
all_letters = string.ascii_letters + " ,.;'"
n_letters = len(all_letters)
print("all_letters-->", all_letters)
print("len-->", len(all_letters))
# 准备所有国家列表
categories = ['Italian',
              'English',
              'Arabic',
              'Spanish',
              'Scottish',
              'Irish',
              'Chinese',
              'Vietnamese',
              'Japanese',
              'French',
              'Greek',
              'Dutch',
              'Korean',
              'Polish',
              'Portuguese',
              'Russian',
              'Czech',
              'German'
              ]
print("categories-->", categories)
category_num = len(categories)
print("len-->", len(categories))


def read_data(filename):
    # 准备两个列表存储人名和国家名
    my_list_x = []
    my_list_y = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            my_list_x.append(line.strip().split('\t')[0])
            my_list_y.append(line.strip().split('\t')[1])
    # print("my_list_x-->", my_list_x)
    # print("my_list_y-->", my_list_y)
    return my_list_x, my_list_y


def test_read_data():
    my_list_x, my_list_y = read_data('../data/name_classfication.txt')


class MyDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y

        # 设定样本长度
        self.sample_len = len(my_list_y)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # 对item进行修正
        item = min(max(item, 0), self.sample_len - 1)
        # 获取x和y
        x = self.my_list_x[item]
        y = self.my_list_y[item]
        # print("x-->", x) # Abl
        # print("y-->", y) # Czech
        # 对x和y进行数值化+张量化
        tensor_x = torch.zeros(len(x), n_letters)
        # print("tensor_x-->", tensor_x, tensor_x.shape) # [3, 57]
        for idx, letter in enumerate(x):
            # print('idx-->', idx,letter)
            tensor_x[idx][all_letters.find(letter)] = 1
        # print("tensor_x-->", tensor_x, tensor_x.shape)
        tensor_y = torch.tensor(categories.index(y))
        # print("tensor_y-->", tensor_y)
        # 返回tensor_x和tensor_y
        return tensor_x, tensor_y


# 自定义RNN
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyRNN, self).__init__()
        self.inputsize = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 定义rnn层
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # 定义全连接层
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        out, hidden = self.rnn(input, hidden)

        result = self.linear(out[:, -1, :])  # [1, 128] -> [128, 18] -> [1, 18]

        return self.softmax(result), hidden

    def init_hidden(self):
        # 手动初始化hidden_size
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 自定义LSTM
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyLSTM, self).__init__()
        self.inputsize = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 定义lstm层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # 定义全连接层
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c0):
        out, (hidden, cn) = self.lstm(input, (hidden, c0))
        result = self.linear(out[:, -1, :])
        return self.softmax(result), hidden, cn

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 自定义GRU
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # 属性
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 定义RNN层
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 定义线性层
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input->[1, 7, 57]，batch_first为True的情况
        # hidden->[1, 1, 128]
        # 把数据送给rnn模型
        output, hidden = self.gru(input, hidden)
        # print("output-->", output.shape)  # [1, 7, 128]
        # print("hn-->", hn.shape)  # [1, 1, 128]
        # 把rnn的输出结果送给线性层
        # 线性层一般接收的是2维数据，所以要把3维降低到2维
        # print("output[:, -1, :]-->", output[:, -1, :].shape)  # [1, 128]
        result = self.linear(output[:, -1, :])  # [1, 128] -> [128, 18] -> [1, 18]
        # print("result-->", result.shape)  # [1, 18]，1个样本，这个样本对应18个类别的概率
        return self.softmax(result), hidden

    def init_hidden(self):
        # 手动初始化hidden_size
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 测试自定义的网络
def test_myrnn():
    # input_size：输入的数据维度，这里就是57
    # hidden_size：RNN输出的数据维度，这里设置为128
    # output_size：最终类别输出，这里就是18分类
    myrnn = MyRNN(input_size=n_letters, hidden_size=128, output_size=category_num)
    # 准备数据
    my_list_x, my_list_y = read_data('../data/name_classfication.txt')
    my_dataset = MyDataset(my_list_x, my_list_y)
    # print('my_dataset-->', my_dataset)
    # 转换为DataLoader：批次拿样本，随机乱序
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 遍历my_dataloader
    for x, y in my_dataloader:
        print('x-->', x, x.shape)  # [1, 4, 57]：1个样本，这个样本有4个token（字母），每个字母用57维向量表示
        print('y-->', y, y.shape)  # [1]
        # 前向传播
        output, hn = myrnn(x, myrnn.init_hidden())
        print("output-->", output.shape)
        print("hn-->", hn.shape)


# TODO 3.模型训练
def train_rnn_model():
    # 准备数据
    my_list_x, my_list_y = read_data('../data/name_classfication.txt')
    my_dataset = MyDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 准备模型
    hidden_size = 128
    model = MyRNN(input_size=n_letters, hidden_size=hidden_size, output_size=category_num)
    # 损失函数，这种写法结合softmax一起，等价于nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    # epoch
    epochs = 1
    # 其他参数，用于打印日志和绘图）
    start_time = time.time()
    total_iter_num = 0  # 纪录训练的总次数
    total_loss = 0  # 总损失值
    total_loss_list = []  # 每隔100轮，平均损失列表
    total_acc = 0  # 总的预测准确数
    total_acc_list = []  # 每隔100轮，平均准确率列表
    # 外层遍历 -> 内层遍历 -> (前向传播，损失计算、梯度清零、反向传播、梯度更新)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # 前向传播
            output, hn = model(x, model.init_hidden())
            # 损失计算
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 纪录数据，用于画图，打印日志
            # 纪录总训练次数
            total_iter_num += 1
            # 纪录总损失
            total_loss += loss.item()
            # 纪录总的准确数
            flag = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc += flag
            # 每隔100个批次，纪录平均损失和准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                # 平均准确率
                avg_acc = total_acc / total_iter_num
                # 添加到列表中
                total_loss_list.append(avg_loss)
                total_acc_list.append(avg_acc)
            # 每隔2000个批次，打印日志
            if total_iter_num % 2000 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                # 平均准确率
                avg_acc = total_acc / total_iter_num
                print("轮次:", epoch + 1, "平均损失:", avg_loss, '平均准确率:', avg_acc, "耗时：",
                      time.time() - start_time)
        # 模型保存
        torch.save(model.state_dict(), '../model/rnn_%d.pth' % (epoch + 1))
        # 保存准确率,损失,耗时等指标,用于绘图
        total_time = time.time() - start_time
        data = {"total_loss_list": total_loss_list, "total_acc_list": total_acc_list, "total_time": total_time}
        # 把data写出
        with open('rnn.json', mode='w', encoding='utf-8') as f:
            f.write(json.dumps(data))


def train_lstm_model():
    # 准备数据
    my_list_x, my_list_y = read_data('../data/name_classfication.txt')
    my_dataset = MyDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 准备模型
    hidden_size = 128
    model = MyLSTM(input_size=n_letters, hidden_size=hidden_size, output_size=category_num)
    # 损失函数，这种写法结合softmax一起，等价于nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    # epoch
    epochs = 1
    # 其他参数，用于打印日志和绘图）
    start_time = time.time()
    total_iter_num = 0  # 纪录训练的总次数
    total_loss = 0  # 总损失值
    total_loss_list = []  # 每隔100轮，平均损失列表
    total_acc = 0  # 总的预测准确数
    total_acc_list = []  # 每隔100轮，平均准确率列表
    # 外层遍历 -> 内层遍历 -> (前向传播，损失计算、梯度清零、反向传播、梯度更新)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # 前向传播
            output, hn, cn = model(x, model.init_hidden(), model.init_hidden())
            # 损失计算
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 纪录数据，用于画图，打印日志
            # 纪录总训练次数
            total_iter_num += 1
            # 纪录总损失
            total_loss += loss.item()
            # 纪录总的准确数
            flag = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc += flag
            # 每隔100个批次，纪录平均损失和准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                # 平均准确率
                avg_acc = total_acc / total_iter_num
                # 添加到列表中
                total_loss_list.append(avg_loss)
                total_acc_list.append(avg_acc)
            # 每隔2000个批次，打印日志
            if total_iter_num % 2000 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                # 平均准确率
                avg_acc = total_acc / total_iter_num
                print("轮次:", epoch + 1, "平均损失:", avg_loss, '平均准确率:', avg_acc, "耗时：",
                      time.time() - start_time)
        # 模型保存
        torch.save(model.state_dict(), '../model/lstm_%d.pth' % (epoch + 1))
        # 保存准确率,损失,耗时等指标,用于绘图
        total_time = time.time() - start_time
        data = {"total_loss_list": total_loss_list, "total_acc_list": total_acc_list, "total_time": total_time}
        # 把data写出
        with open('lstm.json', mode='w', encoding='utf-8') as f:
            f.write(json.dumps(data))


def train_gru_model():
    # 准备数据
    my_list_x, my_list_y = read_data('../data/name_classfication.txt')
    my_dataset = MyDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 准备模型
    hidden_size = 128
    model = MyGRU(input_size=n_letters, hidden_size=hidden_size, output_size=category_num)
    # 损失函数，这种写法结合softmax一起，等价于nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    # epoch
    epochs = 1
    # 其他参数，用于打印日志和绘图）
    start_time = time.time()
    total_iter_num = 0  # 纪录训练的总次数
    total_loss = 0  # 总损失值
    total_loss_list = []  # 每隔100轮，平均损失列表
    total_acc = 0  # 总的预测准确数
    total_acc_list = []  # 每隔100轮，平均准确率列表
    # 外层遍历 -> 内层遍历 -> (前向传播，损失计算、梯度清零、反向传播、梯度更新)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # 前向传播
            output, hn = model(x, model.init_hidden())
            # 损失计算
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 纪录数据，用于画图，打印日志
            # 纪录总训练次数
            total_iter_num += 1
            # 纪录总损失
            total_loss += loss.item()
            # 纪录总的准确数
            flag = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc += flag
            # 每隔100个批次，纪录平均损失和准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                # 平均准确率
                avg_acc = total_acc / total_iter_num
                # 添加到列表中
                total_loss_list.append(avg_loss)
                total_acc_list.append(avg_acc)
            # 每隔2000个批次，打印日志
            if total_iter_num % 2000 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                # 平均准确率
                avg_acc = total_acc / total_iter_num
                print("轮次:", epoch + 1, "平均损失:", avg_loss, '平均准确率:', avg_acc, "耗时：",
                      time.time() - start_time)
        # 模型保存
        torch.save(model.state_dict(), '../model/gru_%d.pth' % (epoch + 1))
        # 保存准确率,损失,耗时等指标,用于绘图
        total_time = time.time() - start_time
        data = {"total_loss_list": total_loss_list, "total_acc_list": total_acc_list, "total_time": total_time}
        # 把data写出
        with open('gru.json', mode='w', encoding='utf-8') as f:
            f.write(json.dumps(data))


# 加载绘图需要的数据
def load_plot_data(filename):
    # 读取数据
    with open(filename, mode='r', encoding='utf-8') as f:
        data = json.loads(f.read())
        # 得到绘图的指标
        total_loss_list = data['total_loss_list']
        total_acc_list = data['total_acc_list']
        total_time = data['total_time']
        # 返回
        return total_loss_list, total_acc_list, total_time


# 绘图函数
def plot_data():
    # 加载rnn的数据
    total_loss_list_rnn, total_acc_list_rnn, total_time_rnn = load_plot_data('rnn.json')
    total_loss_list_lstm, total_acc_list_lstm, total_time_lstm = load_plot_data('lstm.json')
    total_loss_list_gru, total_acc_list_gru, total_time_gru = load_plot_data('gru.json')

    # 绘制损失曲线
    plt.plot(total_loss_list_rnn, label='rnn')
    plt.plot(total_loss_list_lstm, label='lstm')
    plt.plot(total_loss_list_gru, label='gru')

    plt.legend(loc='upper right')
    plt.savefig('../model/loss.png')
    plt.show()
    # 绘制耗时图
    data_x = ['RNN', 'LSTM', 'GRU']
    data_y = [total_time_rnn, total_time_lstm, total_time_gru]
    plt.bar(data_x, data_y)
    plt.savefig('../model/time.png')
    plt.show()
    # 绘制准确率曲线
    plt.plot(total_acc_list_rnn, label='rnn')
    plt.plot(total_acc_list_lstm, label='lstm')
    plt.plot(total_acc_list_gru, label='gru')
    plt.legend(loc='upper left')
    plt.savefig('../model/acc.png')
    plt.show()


# TODO 4.模型预测
# 名字数值化+张量化
def name2tensor(name):
    tensor_x = torch.zeros(len(name), n_letters)
    for i, letter in enumerate(name):
        tensor_x[i][all_letters.find(letter)] = 1

    return tensor_x


# RNN预测
def predict_rnn(name):
    tensor_x = name2tensor(name)
    model = MyRNN(input_size=n_letters, hidden_size=128, output_size=category_num)
    model.load_state_dict(torch.load('../model/rnn_1.pth'))

    with torch.no_grad():
        output, hn = model(tensor_x.unsqueeze(dim=0), model.init_hidden())
        # print("output-->", output.shape)
        # print("output-->", output)
        # 从结果中取前3个国家名
        topv, topi = torch.topk(output, k=3)
        # print("topv-->", topv)  # [1, 3]
        # print("topi-->", topi)  # [1, 3]
        for idx in topi[0]:
            # print(idx)  # 6,1,7
            # 基于index下标得到国家名
            country = categories[idx]
            print("name-->", name, "country-->", country)

# LSTM预测
def predict_lstm(name):
    tensor_x = name2tensor(name)
    model = MyLSTM(input_size=n_letters, hidden_size=128, output_size=category_num)
    model.load_state_dict(torch.load('../model/lstm_1.pth'))

    with torch.no_grad():
        output, hn, cn= model(tensor_x.unsqueeze(dim=0), model.init_hidden(), model.init_hidden())

        topv, topi = torch.topk(output, k=3)
        for idx in topi[0]:
            country = categories[idx]
            print("name-->", name, "country-->", country)


# GRU预测
def predict_gru(name):
    tensor_x = name2tensor(name)
    model = MyGRU(input_size=n_letters, hidden_size=128, output_size=category_num)
    model.load_state_dict(torch.load('../model/gru_1.pth'))

    with torch.no_grad():
        output, hn = model(tensor_x.unsqueeze(dim=0), model.init_hidden())
        # print("output-->", output.shape)
        # print("output-->", output)
        # 从结果中取前3个国家名
        topv, topi = torch.topk(output, k=3)
        # print("topv-->", topv)  # [1, 3]
        # print("topi-->", topi)  # [1, 3]
        for idx in topi[0]:
            # print(idx)  # 6,1,7
            # 基于index下标得到国家名
            country = categories[idx]
            print("name-->", name, "country-->", country)

if __name__ == '__main__':
    # test_read_data()
    # test_myrnn()
    # train_rnn_model()
    # train_lstm_model()
    # train_gru_model()
    # plot_data()
    predict_rnn('Trump')
    predict_lstm('Trump')
    predict_gru('Trump')