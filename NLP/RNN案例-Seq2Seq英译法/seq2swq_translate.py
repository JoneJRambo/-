import re

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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



# TODO 3.模型训练
# TODO 4.模型预测


if __name__ == '__main__':
    # load_data(data_path)
    # test_dataloader()
    # test_encoder()
    test_decoder()