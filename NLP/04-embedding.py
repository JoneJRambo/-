import torch.nn as nn
import torch

# num_embeddings: 词表大小, 也就是词表中一共有多少个token
# embedding_dim: 词向量的维度 特征数
embed = nn.Embedding(10000, 9)

# data = torch.tensor(10000)
# print(data.shape)
# print(embed(data))
data = torch.tensor(9999)
result = embed(data)
print(result)
