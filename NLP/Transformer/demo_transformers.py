import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


# 文本分类
def dm01():
    # 加载模型
    model = AutoModel.from_pretrained("./model/chinese_sentiment")
    # 加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained("./model/chinese_sentiment")

    # 准备文本
    msg = "人生几何对酒当歌"

    # 对文本进行数值化 + 张量化
    # input = my_tokenizer.encode(text=msg, padding=True, truncation=True, max_length=20, return_tensors="pt")
    input = my_tokenizer.encode(text=msg, padding='max_length', truncation=True, max_length=20, return_tensors="pt")

    print("input-->", input)

    output = model(input)

    print("output-->", output)
    # [1, 20, 768]: 1个样本，20个token，每个token使用768维向量表示
    print("output.last_hidden_state-->", output.last_hidden_state.shape)
    # [1, 768]: 1个样本，768维向量表示,句向量
    print("output.pooler_output-->", output.pooler_output.shape)


def dm02():
    # 加载模型
    # model = AutoModel.from_pretrained("./model/chinese_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("./model/chinese_sentiment")
    # 加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained("./model/chinese_sentiment")

    # 准备文本
    msg = "人生几何对酒当歌"

    # 对文本进行数值化 + 张量化
    # input = my_tokenizer.encode(text=msg, padding=True, truncation=True, max_length=20, return_tensors="pt")
    input = my_tokenizer.encode(text=msg, padding='max_length', truncation=True, max_length=20, return_tensors="pt")

    print("input-->", input)

    output = model(input)

    print("output-->", output)
    # [1, 20, 768]: 1个样本，20个token，每个token使用768维向量表示
    # print("output.last_hidden_state-->", output.last_hidden_state.shape)
    # # [1, 768]: 1个样本，768维向量表示,句向量
    # print("output.pooler_output-->", output.pooler_output.shape)


if __name__ == '__main__':
    # dm01()
    dm02()