from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

text = "BERT is a transformers model pretrained on a large corpus of English data " \
       "in a self-supervised fashion. This means it was pretrained on the raw texts " \
       "only, with no humans labelling them in any way (which is why it can use lots " \
       "of publicly available data) with an automatic process to generate inputs and " \
       "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
       "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
       "words in the input then run the entire masked sentence through the model and has " \
       "to predict the masked words. This is different from traditional recurrent neural " \
       "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
       "models like GPT which internally mask the future tokens. It allows the model to learn " \
       "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
       " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
       "sentences that were next to each other in the original text, sometimes not. The model then " \
       "has to predict if the two sentences were following each other or not."

# 1 加载tokenizer
my_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="./model/distilbart-cnn-12-6")

# 2 加载模型
my_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path='./model/distilbart-cnn-12-6')

# 3 文本数值化+张量化
# input = my_tokenizer([text], return_tensors='pt')
# print('input--->', input)
input = my_tokenizer(text, return_tensors='pt')
# print('input--->', input)

# 4 送给模型做摘要
my_model.eval()
output = my_model.generate(input.input_ids)
print('output--->', output)

# 5 处理摘要结果
# 5-1 decode 的 skip_special_tokens 参数可以去除 token 前面的特殊字符
# clean_up_tokenization_spaces：清理token之间的空格（标点符号和文本是挨着的，没有空格）
print([my_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output])

# 5-2 convert_ids_to_tokens 函数只能将 ids 还原为 token
# print(my_tokenizer.convert_ids_to_tokens(output[0]))