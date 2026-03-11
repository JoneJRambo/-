import torch
import torch.nn as nn


class MyAttn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()

        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        self.attn = nn.Linear(query_size + key_size, value_size1)

        self.attn_combine = nn.Linear(query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        attn_weight = torch.softmax(self.attn(torch.cat([Q, K], dim=-1)), dim=-1)
        # print(attn_weight.shape)

        attn_applied = torch.matmul(attn_weight, V)

        output = torch.cat([Q, attn_applied], dim=-1)

        output = self.attn_combine(output)

        return output, attn_weight[0]


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
