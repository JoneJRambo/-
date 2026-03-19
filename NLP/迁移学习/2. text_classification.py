import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer


# 加载预训练模型
bert_model = BertModel.from_pretrained('../Transformer/model/bert-base-chinese')
bert_tokenizer = BertTokenizer.from_pretrained('../Transformer/model/bert-base-chinese')

# TODO 1.数据预处理
def collate_fn1(batch):
    # print("batch", batch)
    # 从batch中,提取文本和标签
    texts = [line['text'] for line in batch]
    labels = [line['label'] for line in batch]
    print("texts-->", texts)
    print("labels-->", labels)
    # 对文本进行数值化+张量化
    input = bert_tokenizer(texts,
                         padding='max_length',
                         truncation=True,
                         max_length=300,
                         return_tensors='pt')
    print("input-->", input)
    # inputids: 输入数据,必须有
    # token_type_ids: 句子类型,可以有
    # attention_mask: 掩码,最好有
    input_ids = input['input_ids']
    token_type_ids = input['token_type_ids']
    attention_mask = input['attention_mask']

    # 对标签进行张量化
    labels = torch.tensor(labels)
    # print("labels-->", labels)

    # 返回结果
    return input_ids,token_type_ids, attention_mask, labels

def load_data():
    my_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # 把dataset转换成dataloader
    # collate_fn: 二次处理函数
    # drop_last: 是否丢弃最后一个batch，默认为False  这里未True 删除最后一个不足批次的样本
    train_dataloader = DataLoader(dataset=my_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn1,
                                  drop_last=True)
    for input_ids,token_type_ids, attention_mask, labels in train_dataloader:
        print("input_ids-->", input_ids)
        print("token_type_ids-->", token_type_ids)
        print("attention_mask-->", attention_mask)
        print("labels-->", labels)
        break


# TODO 2.构建模型
# TODO 3.训练模型
# TODO 4.模型预测

if __name__ == '__main__':
    load_data()
