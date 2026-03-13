import re
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 特殊token
SOS_TOKEN = 0  # 起始
EOS_TOKEN = 1  # 结束

# 句子长度
MAX_LENGTH = 10

# 文本路径
data_path = '../data/eng-fra-v2.txt'


# 使用正则进行数据处理
def normalize_string(s):
    s = s.lower().strip()
    # 使用正则进行句子分组 在组前加空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 将非字母字符替换为空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# TODO 1.数据预处理
# 加载数据
def load_data(filename):
    data = open(filename, mode='r', encoding='utf-8').read().strip().split('\n')
    # print(data)

    # 对文本正则处理
    my_pairs = [[normalize_string(s) for s in l.split('\t')] for l in data]
    # print(my_pairs[9])

    # 构建英文和法文词表
    english_word2index = {"SOS": 0, "EOS": 1}
    english_word_n = len(english_word2index)
    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = len(french_word2index)

    for pair in my_pairs:
        # print(pair)
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # 构建另外2个词表
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}
    # 返回结果
    return english_word2index, english_index2word, english_word_n, \
        french_word2index, french_index2word, french_word_n, my_pairs


# 调用读取数据的函数
english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, my_pairs = load_data(data_path)


# 转换为Dataset
class MyDataset(Dataset):
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # 对index异常值进行修正
        item = min(max(0, item), self.sample_len - 1)
        # 获取 英文x 和 法文 y
        x = self.my_pairs[item][0]
        y = self.my_pairs[item][1]

        # 文本数值化+张量化
        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_TOKEN)
        tensor_x = torch.tensor(x, device=device)
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_TOKEN)
        tensor_y = torch.tensor(y, device=device)

        return tensor_x, tensor_y


# 转为Dataloader
def test_dataloader():
    my_dataset = MyDataset(my_pairs)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
    # for x, y in my_dataloader:
    #     print(x, y)
    for idx, line in enumerate(my_dataloader):
        if idx > 10:
            break
        print(line)


# TODO 2.模型构建
class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        # 定义GRU层
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, hidden):
        # 把数据送给Embedding层
        x = self.embedding(x)

        output, hidden = self.gru(x, hidden)

        return output, hidden

    def init_hidden(self):
        # 初始化hidden
        hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        return hidden


def test_encoder():
    mydataset = MyDataset(my_pairs)
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)

    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256).to(device)
    for x, y in mydataloader:
        # 获取初始hidden
        hidden = encoder.init_hidden()
        # 获取输出
        output, hidden = encoder(x, hidden)
        print(output.shape)
        print(hidden.shape)
        break


class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        # 定义GRU层
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # 定义全连接层, 用于多分类
        self.fc = nn.Linear(hidden_size, vocab_size)
        # 定义softmax层
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)

        # 经过relu激活
        embedded = torch.relu(embedded)

        output, hidden = self.gru(embedded, hidden)

        output = self.fc(output)

        output = self.softmax(output)

        return output, hidden


def test_decoder():
    mydataset = MyDataset(my_pairs)
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256).to(device)
    print("enconder-->", encoder)
    decoder = DecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256).to(device)
    print("decoder-->", decoder)

    for x, y in mydataloader:
        print("x-->", x, x.shape)
        print("y-->", y, y.shape)
        # 编码
        output, hidden = encoder(x, encoder.init_hidden())
        print("output-->", output.shape)
        # 解码:自回归机制
        # 句子: ABCDE
        # 流程 A->B AB->C ABC->D ABCD->E
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
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 定义Embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)
        # 定义attn层
        self.attn = nn.Linear(in_features=input_size + hidden_size, out_features=max_length)
        # 定义attn_combine线性层
        self.attn_combine = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        # 定义gru层
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # 定义out线性层
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):  # Q K V
        # print('input-->', input.shape)
        # print('hidden-->', hidden.shape)
        # print('encoder_outputs-->', encoder_outputs.shape)
        # 经过词嵌入层
        embedded = self.embedding(input)
        # 随机失活
        embedded = self.dropout(embedded)
        # print("embedded-->", embedded.shape)
        # 拼接


def test_attn_decoder():
    mydataset = MyDataset(my_pairs)
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256).to(device)
    print("enconder-->", encoder)
    decoder = AttnDecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256).to(device)
    print("decoder-->", decoder)
    output, hidden = encoder(x, encoder.init_hidden())
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for idx in range(output.shape[1]):
        encoder_output_c[idx] = output[0][idx]
    print("encoder_output_c-->", encoder_output_c.shape)

    for i in range(y.shape[1]):
        tmp = y[0][i].reshape(1, -1)
        print('tmp-->', tmp.shape)
        # 把QKV送给decoder
        decoder_output, hidden, attn_weights = decoder(tmp, hidden, encoder_output_c.unsqueeze(dim=0))
        break


# TODO 3.模型训练
def train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion):
    # 编码
    encoder_output_c, hidden = encoder(x, encoder.init_hidden())
    decoder_output, hidden, attn_weights = decoder(y, hidden, encoder_output_c.unsqueeze(dim=0))
    loss = criterion(decoder_output, y)


def train_seq2seq():
    mydataset = MyDataset(my_pairs)
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)

    # 准备模型
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256).to(device)
    decoder = AttnDecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256).to(device)

    # 损失函数
    criterion = nn.NLLLoss()
    # 学习率
    learning_rate = 1e-4
    # 优化器
    encoder_optim = torch.optim.Adam(params=encoder.parameters(), lr=learning_rate)
    decoder_optim = torch.optim.Adam(params=decoder.parameters(), lr=learning_rate)
    # 轮次
    epochs = 1
    # 其他参数，用于打印日志和绘图）
    plot_loss_list = []
    print_loss_total = 0
    plot_loss_total = 0
    start_time = time.time()
    print_interval_num = 1000
    plot_interval_num = 100

    # 遍历
    for epoch in range(1, epochs + 1):
        for item, (x, y) in enumerate(tqdm(mydataloader), start=1):
            # 封装前向传播的方法,得到损失
            loss = train_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion)
            # print("loss-->", loss)
            print_loss_total += loss
            plot_loss_total += loss

            # 每隔1000个batch打印一次损失
            if item % print_interval_num == 0:
                print_loss_avg = print_loss_total / print_interval_num
                print_loss_total = 0
                print("epoch:", epoch,"loss:", print_loss_avg,"time:", time.time() - start_time)

            if item % plot_interval_num == 0:
                avg_loss = plot_loss_total / plot_interval_num
                plot_loss_list.append(avg_loss)
                plot_loss_total = 0
        torch.save(encoder.state_dict(), "encoder.pth")
        torch.save(decoder.state_dict(), "decoder.pth")

def draw():




# TODO 4.模型预测
def predict_seq2seq():
    encoder = EncoderGRU(vocab_size=english_word_n, input_size=128, hidden_size=256).to(device)
    decoder = AttnDecoderGRU(vocab_size=french_word_n, input_size=128, hidden_size=256).to(device)

    my_pairs = [
        ("i am a boy.", "je suis un garcon."),
        ("i am a girl.", "je suis une fille."),
        ("i am a man.", "je suis un homme."),
        ("i am a woman.", "je suis une femme."),
        ("i am a child.", "je suis un enfant."),
        ("i am a child.", "je suis un enfant."),
     ]

    for idx, pair in enumerate(my_pairs):
        x = pair[0]
        y = pair[1]

        tmpx =


if __name__ == '__main__':
    # load_data(data_path)
    # test_dataloader()
    # test_encoder()
    test_decoder()
