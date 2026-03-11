import torch
import torch.nn as nn


class MyAttn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # 属性
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size
        # 定义两个线性层
        # 第一个线性层:
        # 参数1:接受Q和K拼接后的维度, 即query_size+key_size拼接的维度大小
        # 参数2:输出维度, 即value_size1 这个线性层的输出要能够和V进行矩阵乘法,而的形状时[1,value_size1,value_size2],
        # 根据矩阵乘法规则 这里线性层的输出就是value_size1
        self.attn = nn.Linear(query_size + key_size, value_size1)

        # 第二个线性层
        # 参数1:接受Q拼接注意力计算第二步结果,第二步结果中,形状就是value_size2, 所以加上Q的query_size
        # 参数2:自定义输出维度, 这里的输出维度就是output_size
        self.attn_combine = nn.Linear(query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        # 拼接Q和K
        # 参数1:输入数据
        # 参数2:拼接的维度
        # 参数3:拼接的维度大小
        # 输出结果: [1,1,query_size+key_size]
        attn_weights = torch.cat((Q, K), dim=-1)
        # 线性层
        # 参数1:输入数据
        # 输出结果: [1,1,value_size1]
        attn_weights = torch.softmax(self.attn(attn_weights), dim=-1)

        attn_applied = attn_weights @ V
        output = torch.cat((Q, attn_applied), dim=-1)

        output = self.attn_combine(output)

        return output, attn_weights[0]


if __name__ == '__main__':
    query_size = 32
    key_size = 32
    value_size1 = 32
    value_size2 = 64
    output_size = 32

    Q = torch.randn(1, 1, query_size)
    K = torch.randn(1, 1, key_size)
    V = torch.randn(1, 32, value_size2)

    my_attn = MyAttn(query_size, key_size, value_size1, value_size2, output_size)
    output, attn_weight = my_attn(Q, K, V)
    print('output-->', output)
    print('attn_weight-->', attn_weight)
