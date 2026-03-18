import json
import random
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import re
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设备 cuda就是GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 起始token
SOS_token = 0
# 结束token
EOS_token = 1
# 句子长度
MAX_LENGTH = 10
# 文本路径
filepath = '../data/eng-fra-v2.txt'


# 使用正则进行数据处理
def normalstring(s):
    # 把s字符串转换为小写，并且去掉首尾空白
    s = s.lower().strip()
    # 使用正则进行分组，在组前加个空格
    s = re.sub(r"([.?!])", r" \1", s)
    # 把其他非英文文本内容替换为空格
    s = re.sub(r"[^a-zA-Z.?!]+", r" ", s)
    # 返回结果
    return s


# TODO 1.数据预处理
# 加载数据
def read_data():
    # with open读取
    with open(filepath, mode='r', encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
    # print("lines-->", lines)
    # 对文本进行正则处理
    my_pairs = [[normalstring(s) for s in line.split('\t')] for line in lines]
    # print("my_pairs-->", my_pairs)
    # 获取下标为10个样本
    sample = my_pairs[10]
    # print("sample-->", sample)
    # print("len(my_pairs)-->", len(my_pairs))

    # 构建英文和法文词表
    english_word2index = {"SOS": 0, "EOS": 1}
    english_word_n = 2
    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = 2
    # 遍历数据，构建2个词表
    for pair in my_pairs:
        # print(pair)
        # 获取pair[0]，就是英文，基于空格切割，得到一个个英文单词
        for word in pair[0].split(' '):
            # word，就是英文单词
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        # 获取pair[1]，就是法文，基于空格切割，得到一个个法文单词
        for word in pair[1].split(' '):
            # word，就是法文单词
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1
        # 打印最终的英文和法文词表
    # print("english_word2index-->", english_word2index)
    # print("len(english_word2index)-->", len(english_word2index))
    # print("french_word2index-->", french_word2index)
    # print("len(french_word2index)-->", len(french_word2index))

    # 构建另外2个词表
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}
    # 返回结果
    return english_word2index, english_index2word, english_word_n, \
        french_word2index, french_index2word, french_word_n, my_pairs


# 调用读取数据的函数
english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, my_pairs = read_data()
# print('english_word2index-->', english_word2index)
# print('english_index2word-->', english_index2word)
# print('english_word_n-->', english_word_n)
# print('french_word2index-->', french_word2index)
# print('french_index2word-->', french_index2word)
# print('french_word_n-->', french_word_n)
# print('my_pairs-->', my_pairs[:10])


# 转换为Dataset
class MyDataset(Dataset):
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # 修正item
        item = min(max(0, item), self.sample_len - 1)
        # 打印line，就是一个列表
        # line = self.my_pairs[item]
        # print("line-->", line)
        # 获取x（英文）和y（法文）
        x = self.my_pairs[item][0]
        y = self.my_pairs[item][1]
        # 对x进行文本数值化+张量化
        # print("x-->", x)
        # print("y-->", y)
        x = [english_word2index[word] for word in x.split(" ")]
        x.append(EOS_token)
        # print("x-->", x)
        tensor_x = torch.tensor(x, device=device)
        # print("tensor_x-->", tensor_x)
        # 对y进行文本数值化+张量化
        y = [french_word2index[word] for word in y.split(" ")]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, device=device)
        # 返回tensor_x和tensor_y
        return tensor_x, tensor_y


def test_mydataset():
    # 准备数据，上面已经有了
    # 实例化Dataset对象
    mydataset = MyDataset(my_pairs=my_pairs)
    for idx, (x, y) in enumerate(mydataset):
        # if idx > 10:
        #     break
        # print(idx, line)
        print('x-->', x, x.shape)  # [1, 8]，1句话，这句话有8个token
        print('y-->', y, y.shape)  # [1, 7]，1句话，这句话有7个token


# 转换为DataLoader
def test_dataloader():
    # 初始化Dataset对象
    mydataset = MyDataset(my_pairs=my_pairs)
    # 把Dataset转换为DataLoader
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    for idx, (x, y) in enumerate(mydataloader):
        if idx > 10:
            break
        print(idx, x, y)


# TODO 2.模型构建
class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super().__init__()
        # 属性
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 定义embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        # 定义gru层
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, input, hidden):
        # print("input-->", input.shape)  # [1, 6]，1个样本，这个样本有6个token
        # print("hidden-->", hidden.shape)  # [1, 1, 256]
        # 把数据送给Embedding层
        embedded = self.embedding(input)
        # print("embedded-->", embedded.shape)  # [1, 6, 128]，1个样本，6个token，每个token使用128维向量表示
        # 把Embedding后的数据送给gru层
        output, hidden = self.gru(embedded, hidden)
        # 返回结果
        # print("output-->", output.shape)  # [1, 6, 256]，1个样本，6个token，每个token使用256维向量表示
        # print("hidden-->", hidden.shape)  # [1, 1, 256]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def test_encodergru():
    # 准备数据
    mydataset = MyDataset(my_pairs=my_pairs)
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    # 准备模型
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256)
    # 遍历mydataloader，得到x和y
    for x, y in mydataloader:
        # 把数据送给模型
        output, hidden = encoder(x, encoder.init_hidden())
        print("output-->", output.shape)  # [1, 6, 256]，1个样本，6个token，每个token使用256维向量表示
        print("hidden-->", hidden.shape)  # [1, 1, 256]
        break


class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super().__init__()
        # 属性
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        # 定义gru层
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # 定义线性层，用于多分类
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # print('input-->', input.shape)                      # [1, 1]
        # print('hidden-->', hidden.shape)                    # [1, 1, 256]
        # 让input输入数据经过Embedding层
        embedded = self.embedding(input)
        # print("embedded0-->", embedded.shape)               # [1, 1, 128]
        # 经过relu激活函数
        embedded = torch.relu(embedded)
        # print("embedded1-->", embedded.shape)               # [1, 1, 128]
        # 把激活后的数据送给gru模型
        output, hidden = self.gru(embedded, hidden)
        print('output-->', output.shape)  # [1, 1, 256]
        # 把output的结果送给线性层
        output = self.softmax(self.out(output[0]))
        # print('output-->', output.shape)                    # [1, 4345]
        # 返回
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def test_decodergru():
    # 准备数据
    mydataset = MyDataset(my_pairs=my_pairs)
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    # 准备模型
    # 编码器模型
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256).to(device)
    print("encoder-->", encoder)
    #  解码器模型
    decoder = DecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256).to(device)
    print("decoder-->", decoder)
    # 让数据先经过编码器（编码），再经过解码器（解码）
    for x, y in mydataloader:
        # 编码
        print("x-->", x, x.shape)
        print("y-->", y, y.shape)
        output, hidden = encoder(x, encoder.init_hidden())
        print("output-->", output.shape)  # [1, 5, 256]
        print("hidden-->", hidden.shape)  # [1, 1, 256]
        # 解码：自回归机制，一个token一个token解码
        # 句子：ABCDE
        # 流程：A->B，AB->C，ABC->D,ABCD->E
        for i in range(y.shape[1]):
            # 构建解码的第一个输入token
            tmp = y[0][i].reshape(1, -1)
            print('tmp-->', tmp.shape)  # [1, 1]
            # 把tmp送给解码器
            # decoder(tmp, decoder.init_hidden())
            decoder_output, hidden = decoder(tmp, decoder.init_hidden())
            print("decoder_output-->", decoder_output.shape)  # [1, 4345]
            print("hidden-->", hidden.shape)
            break
        break


class AttnDecoderGRU(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        # 词表大小
        self.vocab_size = vocab_size
        # 输入数据的词向量维度
        self.input_size = input_size
        # GRU模型输出的隐藏层维度
        self.hidden_size = hidden_size
        # 随机失活概率
        self.dropout_p = dropout_p
        # 模型最大输出
        self.max_length = max_length
        # 定义Embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)
        # 定义attn线性层
        self.attn = nn.Linear(in_features=input_size + hidden_size, out_features=max_length)
        # 定义attn_combine线性层
        self.attn_combine = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        # 定义gru层
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # 定义out线性层
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_output):
        # print("input-->", input.shape)
        # print("hidden-->", hidden.shape)
        # print("encoder_output-->", encoder_output.shape)
        # 把数据送给Embedding层
        embedded = self.embedding(input)
        # print("embedded0-->", embedded.shape)
        # 经过dropout层
        embedded = self.dropout(embedded)
        # print("embedded1-->", embedded.shape)
        # embedded和hidden拼接，送给线性层，经过softmax，得到注意力权重
        attn_weight = torch.softmax(self.attn(torch.cat([embedded, hidden], dim=-1)), dim=-1)
        # print("attn_weight-->", attn_weight.shape)
        # 把注意力权重和V相乘，得到注意力表示
        attn_applied = torch.matmul(attn_weight, encoder_output)
        # print("attn_applied-->", attn_applied.shape)            # [1, 1, 256]
        # 把attn_applied和embedded再次拼接
        embedded_combine = torch.cat([attn_applied, embedded], dim=-1)
        # print("embedded_combine-->", embedded_combine.shape)    # [1, 1, 384]
        # 把拼接后的结果再次送给线性层
        embedded_output = self.attn_combine(embedded_combine)
        # print("embedded_output0-->", embedded_output.shape)     # [1, 1, 256]
        # 经过relu激活函数
        embedded_output = torch.relu(embedded_output)
        # print("embedded_output1-->", embedded_output.shape)     # [1, 1, 256]
        # 把结果送给GRU模型
        decoder_output, hidden = self.gru(embedded_output, hidden)
        # print("decoder_output-->", decoder_output.shape)         # [1, 1, 256]
        # print("hidden-->", hidden.shape)         # [1, 1, 256]
        # 把decoder_output经过线性层，得到最终分类
        output = self.out(decoder_output[0])
        # print("output0-->", output.shape)                         # [1, 4345]
        output = self.softmax(output)
        # print("output1-->", output.shape)                         # [1, 4345]
        # 返回结果
        return output, hidden, attn_weight


def test_attndecoder():
    mydataset = MyDataset(my_pairs=my_pairs)
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256).to(device)
    print("encoder-->", encoder)
    #  解码器模型
    decoder = AttnDecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256).to(device)
    print("decoder-->", decoder)
    # 让数据先经过编码器（编码），再经过解码器（解码）
    for x, y in mydataloader:
        # 编码
        print("x-->", x, x.shape)  # [1, 5]
        print("y-->", y, y.shape)  # [1, 6]
        output, hidden = encoder(x, encoder.init_hidden())
        print("output-->", output.shape)  # [1, 5, 256]
        print("hidden-->", hidden.shape)  # [1, 1, 256]
        # 解码，带注意力的，需要QKV都参与（中间语言张量C也要参与）
        # 自回归机制，由上文解码下文。
        # Q：[1,1]
        # K：hidden
        # V：自己构建
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        # 给V赋值
        for idx in range(output.shape[1]):
            # 把output输出的每一个token的结果都赋值给encoder_output_c
            encoder_output_c[idx] = output[0][idx]
        print("encoder_output_c-->", encoder_output_c)

        for i in range(y.shape[1]):
            # 从y（法文）中取1个进行解码
            tmp = y[0][i].reshape(1, -1)
            print("tmp-->", tmp.shape)
            # 把QKV送给解码器
            output, hidden, attn_weight = decoder(tmp, hidden, encoder_output_c.unsqueeze(dim=0))
            print("output-->", output.shape)
            print("hidden-->", hidden.shape)
            print("attn_weight-->", attn_weight.shape)
            break
        break


# TODO 3.模型训练
def train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion):
    # print("x-->", x, x.shape)
    # print("y-->", y, y.shape)
    # 编码
    encoder_output, encoder_hidden = encoder(x, encoder.init_hidden())
    # print("encoder_output-->", encoder_output.shape)
    # print("encoder_hidden-->", encoder_hidden.shape)
    # 解码
    # Q：input，就是SOS_token
    tensor_x = torch.tensor([[SOS_token]], device=device)
    # print("tensor_x-->", tensor_x.shape)  # [1, 1]
    # K：就是encoder_hidden
    decoder_hidden = encoder_hidden
    # V：自己构建
    encoder_output_c = torch.zeros(MAX_LENGTH, decoder.hidden_size, device=device)  # [10, 256]
    # 给encoder_output_c赋值
    for idx in range(encoder_output.shape[1]):
        encoder_output_c[idx] = encoder_output[0][idx]
    # print("encoder_output_c-->", encoder_output_c)

    # 定义是否使用teacher_forcing策略
    use_teacher_forcing = True if random.random() < 1 else False
    y_len = y.shape[1]
    loss = 0
    if use_teacher_forcing:
        # 走teacher_forcing策略
        # 因为这里是模型训练，训练过程中是知道真是答案的。
        for i in range(y_len):
            # 解码器开始逐个token解码（前向传播）
            decoder_output, decoder_hidden, attn_weights = decoder(tensor_x, decoder_hidden, encoder_output_c)
            print("decoder_output-->", decoder_output, decoder_output.shape)
            # 计算损失
            target_y = y[0][i].reshape(1)
            print("target_y-->", target_y, target_y.shape)
            loss += criterion(decoder_output, target_y)
            # print("loss1-->", loss)
            # 把真确答案给到模型，进行下一轮的训练
            tensor_x = y[0][i].reshape(1, 1)
            break
    else:
        # 不走teacher_forcing策略
        for i in range(y_len):
            decoder_output, decoder_hidden, attn_weights = decoder(tensor_x, decoder_hidden, encoder_output_c)
            # 计算损失
            target_y = y[0][i].reshape(1)
            loss += criterion(decoder_output, target_y)
            # print("loss2-->", loss)
            # 把预测结果给到下一个token
            topv, topi = torch.topk(decoder_output, k=1)
            # print("topv-->", topv)
            # print("topi-->", topi)
            # topv：具体的值
            # topi：下标索引
            # 判断，topi的下标索引是否是结束标记，如果是，则退出。如果不是，则继续下一步预测
            if topi.item() == EOS_token:
                break
            # 给预测结果
            tensor_x = topi.detach()
    # 梯度清零
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    # 反向传播
    loss.backward()
    # 梯度更新
    encoder_optim.step()
    decoder_optim.step()
    # 返回平均loss
    return loss.item() / y_len


def train_seq2seq():
    # 准备数据
    mydataset = MyDataset(my_pairs=my_pairs)
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    # 准备模型
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256)
    decoder = AttnDecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256)
    # 把模型放到GPU上
    encoder.to(device=device)
    decoder.to(device=device)
    # 损失函数
    criterion = nn.NLLLoss()
    # 学习率
    mylr = 1e-4
    # 优化器
    encoder_optim = optim.Adam(params=encoder.parameters(), lr=mylr)
    decoder_optim = optim.Adam(params=decoder.parameters(), lr=mylr)
    # 轮次
    epochs = 20
    # 参数
    plot_loss_list = []
    print_loss_total = 0
    plot_loss_total = 0
    print_interval_num = 1000
    plot_interval_num = 100
    start_time = time.time()
    # 遍历
    for epoch in range(1, epochs + 1):
        for item, (x, y) in enumerate(tqdm(mydataloader), start=1):
            # 把数据放到GPU上
            x.to(device=device)
            y.to(device=device)
            # 封装前向传播的方法，得到损失
            loss = train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion)
            # print("loss-->", loss)
            # 纪录loss
            print_loss_total += loss
            plot_loss_total += loss
            # 打印日志
            if item % print_interval_num == 0:
                avg_loss = print_loss_total / print_interval_num
                print("epoch:", epoch, 'loss:', avg_loss, 'time:', time.time() - start_time)
                print_loss_total = 0
            # 画图
            if item % plot_interval_num == 0:
                avg_loss = plot_loss_total / plot_interval_num
                plot_loss_list.append(avg_loss)
                plot_loss_total = 0
        # 保存模型
        torch.save(encoder.state_dict(), './model/encoder_%d.pth' % epoch)
        torch.save(decoder.state_dict(), './model/decoder_%d.pth' % epoch)
    # 把loss保存为json文件，方便后续画图
    with open('loss.json', mode='w', encoding='utf-8') as f:
        data = {"plot_loss_list": plot_loss_list}
        f.write(json.dumps(data))


# 绘图
def draw():
    # 加载loss.json数据
    with open('loss.json', mode='r', encoding='utf-8') as f:
        data = json.loads(f.read())
    print("data-->", data)
    # 从json中获取损失值
    plot_loss_list = data['plot_loss_list']
    plt.plot(plot_loss_list)
    plt.savefig("./model/loss.png")
    plt.show()


# TODO 4.模型预测
def seq2seq_evaluate(x, encoder, decoder):
    # 编码
    encoder_output, encoder_hidden = encoder(x, encoder.init_hidden())
    # 解码
    # Q：SOS_token
    input_y = torch.tensor([[SOS_token]], device=device)
    # K：encoder_hidden
    decoder_hidden = encoder_hidden
    # V：自己构建
    encoder_output_c = torch.zeros(MAX_LENGTH, decoder.hidden_size, device=device)
    # 给encoder_output_c赋值
    for idx in range(encoder_output.shape[1]):
        encoder_output_c[idx] = encoder_output[0][idx]
    # 自回归机制开始解码
    # 准备2个变量，接收解码的数据
    decoder_words = []
    decoder_attn = torch.zeros(MAX_LENGTH, MAX_LENGTH)  # [10，10]
    # 因为这里是预测，因此我们给模型的最大输出长度，当然可以提前结束
    for i in range(MAX_LENGTH):
        # 基于SOS_token预测结果
        decoder_output, decoder_hidden, attn_weights = decoder(input_y, decoder_hidden, encoder_output_c)
        # 给注意力权重赋值
        # print("attn_weights-->", attn_weights.shape)
        # print("decoder_attn-->", decoder_attn.shape)
        decoder_attn[i] = attn_weights[0][0]
        # 拿到预测结果
        topv, topi = torch.topk(decoder_output, k=1)
        # 判断是否是结束标记
        if topi.item() == EOS_token:
            decoder_words.append("<EOS>")
            break
        else:
            # 输出的不是结束符，需要纪录解码的结果
            decoder_words.append(french_index2word[topi.item()])
        # 把预测结果给下一步的输入
        input_y = topi.detach()
    # 返回
    return decoder_words, decoder_attn


def predict_seq2seq():
    # 准备模型
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256)
    encoder.load_state_dict(torch.load('./model/encoder_20.pth'))
    encoder.to(device=device)
    decoder = AttnDecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256)
    decoder.load_state_dict(torch.load('./model/decoder_20.pth'))
    decoder.to(device=device)
    # 准备数据
    my_pairs = [
        ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
        ['i m more than a friend .', 'je suis plus qu une amie .'],
        ['she is beautiful like her mother .', 'elle est belle comme sa mere .'],
        ['i like music', 'je aime la musique']
    ]
    # 遍历
    for idx, pair in enumerate(my_pairs):
        # print('idx-->', idx, "pair-->", pair)
        # 从pair中获取x和y
        x = pair[0]
        y = pair[1]
        # 对x进行数值化和张量化
        tmpx = [english_word2index[word] for word in x.split(" ")]
        tmpx.append(EOS_token)
        # print("tmpx-->", tmpx)
        tensor_x = torch.tensor(tmpx, device=device).reshape(1, -1)
        # print("tensor_x-->", tensor_x, tensor_x.shape)          # [1, 8]
        # 把数据和模型送给解码函数
        decoder_words, decoder_attn = seq2seq_evaluate(tensor_x, encoder, decoder)
        # print("decoder_words-->", decoder_words)
        # print("decoder_attn-->", decoder_attn)
        print("x-->", x)
        print("y-->", y)
        print("y_pred-->", ' '.join(decoder_words))


if __name__ == '__main__':
    # read_data()
    # test_mydataset()
    # test_dataloader()
    # test_encodergru()
    # test_decodergru()
    # test_attndecoder()
    # train_seq2seq()
    # draw()
    predict_seq2seq()

