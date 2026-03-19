import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


# 加载模型和分词器
model = AutoModelForMaskedLM.from_pretrained("./model/chinese-bert-wwm")
tokenizer = AutoTokenizer.from_pretrained("./model/chinese-bert-wwm")

# 准备文本
msg = "我想明天去[MASK]边钓鱼"

# 把文本送给模型
input = tokenizer._encode_plus(text=msg, return_tensors="pt")
print("input-->", input)
# 解码,得到中文
model.eval()
output = model(**input)
print("output-->", output)
print('output.logits--->', output.logits.shape)

mask_token = torch.argmax(output.logits[0][6])
print("mask_token-->", mask_token)
result = tokenizer.convert_ids_to_tokens(mask_token.item())
print("result-->", result)