from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# 加载模型
# AutoModel（基础模型）的输出包含 last_hidden_state（词向量）、pooler_output（句向量）等属性
# model = AutoModel.from_pretrained('./model/chinese_sentiment')
# AutoModelForSequenceClassification（分类模型）的输出是 SequenceClassifierOutput 对象，
# 核心属性是 logits（分类得分），没有 last_hidden_state 和 pooler_output
model = AutoModelForSequenceClassification.from_pretrained('./model/chinese_sentiment')
# 加载分词器（把数据处理成模型能识别的样子）
tokenizer = AutoTokenizer.from_pretrained("./model/chinese_sentiment")

# 准备文本
message = '我不喜欢上海，房价太贵了'

# 对文本进行数值化 + 张量化
input = tokenizer.encode(text=message, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
print("input-->", input)
# 把文本送给模型，得到结果
output = model(input)
print("output-->", output)
# [1, 20, 768]：1句话，有20个token，每个token768维，每个token的向量都有。词向量
# print("output-->", output.last_hidden_state.shape)
# [1, 768]：1句话，这句话使用一个768维向量表示，句向量
# print("output-->", output.pooler_output.shape)