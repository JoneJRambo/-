import joblib
from tensorflow.keras.preprocessing.text import Tokenizer


def dm01():
    vocabs = {"周杰伦","王力宏","张学友","周华健","陈奕迅"}
    # 创建一个tokenizer分词器
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocabs)
    print('word_index -->',tokenizer.word_index)
    print('index_word -->',tokenizer.index_word)
    for vocab in vocabs:
        # 准备一个全零列表
        zero_list =[0] * len(vocabs)
        idx = tokenizer.word_index[vocab] - 1
        print("idx-->", idx)

        #设置数值1 ,即 one-hot 编码
        zero_list[idx] = 1
        print(vocab,zero_list)
    # 保存tokenizer
    joblib.dump(tokenizer,'./tokenizer')

def dm02():
    tokenizer = joblib.load('./tokenizer')
    print('word_index -->',tokenizer.word_index)
    print('index_word -->',tokenizer.index_word)
    token = '周杰伦'
    # 得到token的下标
    idx = tokenizer.word_index[token] - 1
    # 基于下标,构建 one-hot 编码
    zero_list = [0] * len(tokenizer.word_index)
    print("zero_list-->", zero_list)

    #把token的位置下标设置为1
    zero_list[idx] = 1
    print("zero_list-->",zero_list)








if __name__ == '__main__':
    # dm01()
    dm02()