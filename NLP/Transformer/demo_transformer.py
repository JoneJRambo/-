import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

"""
Tansformer由四部分组成:
1.输入部分
2.编码器部分
3.解码器部分
4.输出部分
"""


# TODO 1.输入部分
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # 属性
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 定义embedding层
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: 为输入的样本  为了保持和positional encoding量纲差不多, 乘以根号d_model
        return self.embedding(x) * math.sqrt(self.d_model)


# 测试输入部分
def test_embedding():
    vocab_size = 1000
    d_model = 512
    embedding = Embeddings(vocab_size, d_model)
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    embedded = embedding(x)
    print("embedded-->", embedded)
    print("embedded.shape-->", embedded.shape)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        # 定义位置编码矩
        self.pe = torch.zeros(max_len, d_model)
        # 准备position位置
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 直接套公式
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float)
        self.pe[:, ::2] = torch.sin(position / torch.pow(10000, _2i / d_model))
        self.pe[:, 1::2] = torch.cos(position / torch.pow(10000, _2i / d_model))
        # 对pe进行升维
        self.pe = self.pe.unsqueeze(0)
        # print("pe-->", self.pe.shape)  # [1, 100, 512]

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]  # [2,4,512] pe[:,:x.shape[1]]等价于pe[:,:x.shape[1],:]
        return self.dropout(x)


def test_positional_encoding():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    test_embedding = Embeddings(vocab_size=1000, d_model=512)
    embedded = test_embedding(x)
    print("embedded-->", embedded.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(embedded)
    print("pe_result-->", pe_result.shape)


# 画图,位置编码
def plot_positional_encoding():
    # 创建pe对象
    pe = PositionalEncoding(d_model=20, dropout=0, max_len=100)
    # 创建输入
    x = torch.zeros(1, 100, 20)
    y = pe(x)
    # 画图
    plt.plot(torch.arange(0, 100), y[0, :, 4:8])
    plt.legend(['dim_%d' % p for p in [4, 5, 6, 7]], loc='best', fontsize='small')
    plt.show()


# TODO 2.编码器部分
def subquent_mask(size):
    # 创建上三角矩阵
    mask = np.triu(m=np.ones((size, size)), k=1).astype('uint8')

    result = torch.from_numpy(1 - mask)

    return result


def test_subquent_mask():
    size = 20
    mask = subquent_mask(size)
    print("mask-->", mask)

    # 画图
    plt.imshow(mask)
    plt.show()


def attention(query, key, value, mask=None, dropout=None):
    """
    :param query: [batch_size, seq_len, d_k]
    :param key: [batch_size, seq_len, d_k]
    :param value: [batch_size, seq_len, d_k]
    :param mask: [batch_size, seq_len, seq_len]
    :param dropout:
    :return: output(输出张量),attn_weights(注意力权重)
    """
    # 得到d_k (词嵌入的维度)
    d_k = query.size(-1)
    # 自注意力公式
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("score-->", score, score.shape)
    # 判断是否使用mask
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
        # print("score-->", score, score.shape)
    # 经过softmax得到注意力权重
    p_attn = torch.softmax(score, dim=-1)
    # 判断是否dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 和V相乘得到注意力输出
    return torch.matmul(p_attn, value), p_attn


def test_attention():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("pe_result-->", pe_result.shape)
    query = key = value = pe_result
    mask = torch.zeros(2, 4, 4)
    # print("mask-->", mask)
    attn, p_attn = attention(query=query, key=key, value=value, mask=mask)
    print("attn-->", attn.shape)
    attn, p_attn = attention(query, key, value)
    print("attn-->", attn.shape)


# clone
def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 多头注意力
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embed_dim, dropout=0.1):
        super().__init__()
        # 断言,保证维度能整除head
        assert embed_dim % head == 0
        # 获取d_k
        self.d_k = embed_dim // head
        self.head = head
        self.attn = None
        self.embed_dim = embed_dim
        # 创建四个线性层
        self.linears = clone(nn.Linear(in_features=embed_dim, out_features=embed_dim), 4)
        # 创建一个dropout层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 对mask进行升维
        if mask is not None:
            mask = mask.unsqueeze(0)
        # 得到batch_size
        batch_size = query.size(0)
        # 推导式实现把数据送给模型,经过线性变换
        # 对输入的 query、key、value 分别应用对应的线性变换（self.linears 包含三个线性层），
        # 将结果重塑为 (batch_size, seq_len, num_heads, head_dim) 的形状，
        # 然后交换第1维（seq_len）和第2维（num_heads），得到 (batch_size, num_heads, seq_len, head_dim)，
        # 以便在多头注意力机制中并行计算。
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose_(1, 2) for model, x in
                             zip(self.linears, (query, key, value))]
        print("query-->", query.shape)
        print("key-->", key.shape)
        print("value-->", value.shape)
        # 执行多头注意力并行计算
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        print("x0-->", x.shape)
        # 对x进行维度变换
        # x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        print("x1-->", x.shape)
        # 通过线性层
        x = self.linears[-1](x)  # clone中最后一个线性层
        return x


def test_multi_headed_attention():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    print('embedded-->', x.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("pe_result-->", pe_result.shape)
    # 把pe_result  送给 Attention注意力函数
    query = key = value = pe_result
    mha = MultiHeadedAttention(head=8, embed_dim=512)
    mask = torch.zeros(8, 4, 4)
    x = mha(query, key, value, mask=mask)
    print("x2-->", x.shape)


# 逐位前馈网络层
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 定义两个线性层
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

        # 定义一个dropout层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 通过第一个线性层
        x = self.w_1(x)
        # 激活函数
        x = torch.relu(x)
        # 添加dropout
        x = self.dropout(x)
        # 通过第二个线性层
        x = self.w_2(x)
        return x


def test_positionwise_feed_forward():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    print('embedded-->', x.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("pe_result-->", pe_result.shape)
    query = key = value = pe_result
    mha = MultiHeadedAttention(head=8, embed_dim=512)
    # 把pe_result  送给 Attention注意力函数
    mask = torch.zeros(8, 4, 4)
    x = mha(query, key, value, mask=mask)
    print("x-->", x.shape)
    # 把x送给前馈层
    # 论文中的d_ff=2048,通过扩展维度再压缩，让模型学习更复杂的特征变换
    pwff = PositionWiseFeedForward(d_model=512, d_ff=2048)
    pwff_result = pwff(x)
    print("pwff_result-->", pwff_result.shape)


# 规范化层
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 准备k和b
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 获取维度
        mean = x.mean(-1, keepdim=True)  # 保持维度
        std = x.std(-1, keepdim=True)
        # 进行归一化
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def test_layer_normalization():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    print("embedded-->", x.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("pe_result-->", pe_result.shape)
    query = key = value = pe_result
    head = 8
    mask = torch.zeros(head, 4, 4)
    multi_headed_attention = MultiHeadedAttention(head=head, embed_dim=512)
    x = multi_headed_attention(query=query, key=key, value=value, mask=mask)
    print("x-->", x.shape)
    # 把x送给前馈层
    pwff = PositionWiseFeedForward(d_model=512, d_ff=2048)
    pwff_result = pwff(x)
    print("pwff_result-->", pwff_result.shape)
    # 创建规范化层
    ln = LayerNormalization(features=512)
    ln_y = ln(pwff_result)
    print("ln_y-->", ln_y.shape)


# 子层连接层
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        # 创建一个dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个层归一化层
        self.norm = LayerNormalization(size)

    def forward(self, x, sublayer):
        # 让x数据先经过子层对象（多头自注意力、前馈全连接），再经过规范化层，最后经过随机失活，然后加上x（残差）
        # 标准Pre-LN 先归一化再子层
        return x + self.dropout(sublayer(self.norm(x)))


def test_sublayer_connection():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    print("embedded-->", x.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("pe_result-->", pe_result.shape)
    query = key = value = pe_result
    # 把pe_result送给Attention注意力函数
    mask = torch.zeros(8, 4, 4)
    mha = MultiHeadedAttention(head=8, embed_dim=512)
    # 构建子层连接结构对象
    sc = SublayerConnection(size=512, dropout=0.1)
    sublayer = lambda x: mha(query, key, value, mask=mask)
    sublayer_x = sc(x, sublayer)
    print("sublayer_x-->", sublayer_x.shape)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        # 创建一个多头注意力层
        self.size = size
        self.self_attn = self_attn
        # 创建一个前馈层
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 创建两个子层连接层
        self.sublayer = clone(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # 通过第一个子层连接层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 通过第二个子层连接层
        return self.sublayer[1](x, self.feed_forward)  # 等价写法: self.sublayer[1](x, lambda x: self.feed_forward(x))


def test_encoder_layer():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    print("embedded-->", x.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("pe_result-->", pe_result.shape)
    query = key = value = pe_result
    mask = torch.zeros(8, 4, 4)
    mha = MultiHeadedAttention(head=8, embed_dim=512)
    # self_attn = lambda x: mha(query, key, value, mask=mask)
    feed_forward = PositionWiseFeedForward(d_model=512, d_ff=2048)
    encoder_layer = EncoderLayer(size=512, self_attn=mha, feed_forward=feed_forward, dropout=0.1)
    encoder_layer_result = encoder_layer(x, mask)
    print("encoder_layer_result-->", encoder_layer_result.shape)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # 构建多个编码器层
        self.layers = clone(layer, N)
        # 规范化层
        self.norm = LayerNormalization(features=layer.size)

    def forward(self, x, mask):
        # 循环调用多个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        # 经过规范化层后返回
        return self.norm(x)


def test_encoder():
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    print("embedded-->", x.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("pe_result-->", pe_result.shape)
    query = key = value = pe_result
    mask = torch.zeros(8, 4, 4)
    mha = MultiHeadedAttention(head=8, embed_dim=512)
    pwff = PositionWiseFeedForward(d_model=512, d_ff=2048)
    # 定义深拷贝 防止所有层共享同一个mha和pwff对象,共用同一套权重参数
    c = copy.deepcopy
    # 构建编码器层对象
    encoder_layer = EncoderLayer(size=512, self_attn=c(mha), feed_forward=c(pwff), dropout=0.1)
    # 构建编码器对象
    encoder = Encoder(layer=encoder_layer, N=6)
    print(encoder)
    # 编码器对象调用
    encoder_result = encoder(x, mask)
    print("encoder_result-->", encoder_result.shape)
    # 返回编码器结果
    return encoder_result


# TODO 3.解码器部分
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        # 创建一个多头自注意力层
        self.size = size
        self.self_attn = self_attn
        # 创建一个交叉多头自注意力层
        self.src_attn = src_attn
        # 创建一个前馈层
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 创建三个子层连接层
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: 编码器的输入
        # memory: 编码器的输出
        # src_mask: 编码器的掩码
        # tgt_mask: 解码器的掩码
        # 通过第一个子层连接层
        m = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 通过第二个子层连接层
        n = self.sublayer[1](m, lambda x: self.src_attn(x, memory, memory, src_mask))
        # 通过第三个子层连接层
        return self.sublayer[2](n, self.feed_forward)


def test_decoder_layer():
    # 调用测试编码器层的结果
    memory = test_encoder()
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    print("decoder_x-->", x, x.shape)
    embedding = Embeddings(vocab_size=1000, d_model=512)
    x = embedding(x)
    print("embedded-->", x.shape)
    pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
    pe_result = pe(x)
    print("decoder_pe_result-->", pe_result.shape)
    query = key = value = pe_result
    src_mask = tgt_mask = mask = torch.zeros(8, 4, 4)
    self_attn = src_attn = mha = MultiHeadedAttention(head=8, embed_dim=512)
    pwff = PositionWiseFeedForward(d_model=512, d_ff=2048)
    # 创建解码器层对象
    decoder_layer = DecoderLayer(size=512, self_attn=self_attn, src_attn=src_attn, feed_forward=pwff, dropout=0.1)
    # 加载数据到解码器
    decoder_layer_result = decoder_layer(x, memory, src_mask, tgt_mask)
    print("decoder_layer_result-->", decoder_layer_result.shape)


# TODO 4.输出部分


if __name__ == '__main__':
    # test_embedding()
    # test_positional_encoding()
    # plot_positional_encoding()
    # test_subquent_mask()
    # test_attention()
    # test_multi_headed_attention()
    # test_positionwise_feed_forward()
    # test_layer_normalization()
    # test_sublayer_connection()
    # test_encoder_layer()
    # test_encoder()
    test_decoder_layer()