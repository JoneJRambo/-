from transformers import AutoModel, AutoTokenizer


model = AutoModel.from_pretrained("./model/bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("./model/bert-base-chinese")

msg =['你是谁','人生该如何起头']

#对文本进行数值化张量化
input = tokenizer(text=msg, padding='max_length', truncation=True, max_length=20, return_tensors="pt")
print('input-->', input)
# input_ids: 文本数值化+张量化结果
print("input_ids",input['input_ids'])
# token_type_ids: 文本的token类型
print("token_type_ids",input['token_type_ids'])
# attention_mask: 文本的attention_mask, 用于BERT模型
print("attention_mask",input['attention_mask'])

model.eval()
output = model(**input)
print('output-->', output.last_hidden_state.shape)
print('output-->', output.pooler_output.shape)