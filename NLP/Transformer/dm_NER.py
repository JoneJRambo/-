import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

model = AutoModelForTokenClassification.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
tokenizer = AutoTokenizer.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
config = AutoConfig.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
# print("config-->", config)
msg = "我觉得杭州西湖还不错"
input = tokenizer(text=msg, padding='max_length', truncation=True, max_length=20, return_tensors="pt")
print('input-->', input)

logits = model(**input).logits
print('logits-->', logits.shape)
# 解码
input_tokens = tokenizer.convert_ids_to_tokens(input.input_ids[0])
# ['[CLS]', '我', '觉', '得', '杭', '州', '西', '湖', '还', '不', '错', '[SEP]']
print("input_tokens-->", input_tokens)
print("tokenizer.all_special_tokens-->", tokenizer.all_special_tokens)

# 遍历，得到每个token的NER
for token, value in zip(input_tokens, logits[0]):
    # 过滤特殊token
    if token in tokenizer.all_special_tokens:
        continue
    # token：每一个token，比如CLS、我等都是一个token
    # value：每一个token对应的32维向量
    # print(token, value)
    idx = torch.argmax(value).item()
    # print("idx-->", idx)
    # 基于idx，换成NER的类别（32个类别中的某个类别）
    # labels = config.id2label
    # print("labels-->", labels)
    label = config.id2label[idx]
    print("token-->", token, "label-->", label)
