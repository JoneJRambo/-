from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# 加载模型和分词器
model = AutoModelForQuestionAnswering.from_pretrained("./model/chinese_pretrain_mrc_roberta_wwm_ext_large")
tokenizer = AutoTokenizer.from_pretrained("./model/chinese_pretrain_mrc_roberta_wwm_ext_large")

# 准备文本数据
context = '我叫张三 我是一个程序员 我的喜好是打篮球'
questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']

# 遍历问题，把问题一个个送给模型
model.eval()
for question in questions:
    input = tokenizer._encode_plus(question ,context, return_tensors='pt')
    print("input-->", input)

    # 把数据送给模型
    output = model(**input)
    # print("output-->", output)
    # print("start-->", output.start_logits.shape)  # [1, 26]
    # print("end-->", output.end_logits.shape)  # [1, 26]
    start = torch.argmax(output.start_logits)
    end = torch.argmax(output.end_logits)
    # print("start-->", start)
    # print("end-->", end)
    # 把数字下标解码成中文
    result = tokenizer.convert_ids_to_tokens(input['input_ids'][0][start: end + 1])
    print("result-->", result)

