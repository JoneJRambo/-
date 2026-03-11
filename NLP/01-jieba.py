import torch
import jieba
from numpy.ma.core import shape


# 精确模式
def dm01():
    text = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能。"
    # cut: 分词
    # cut_all: False, 表示精确模式
    result = jieba.cut(text, cut_all=False)
    print(result)
    print(list(result))
    result = jieba.lcut(text, cut_all=False)
    print(result)


# 全模式： 在精确模式的基础上，对长词进行切分 cut_all: True
def dm02():
    text = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能。"
    # cut: 分词
    # cut_all: True, 默认为False，表示精确模式
    result = jieba.cut(text, cut_all=True)
    print(result)
    result = jieba.lcut(text, cut_all=True)
    print(result)


# 搜索引擎模式： 搜索引擎模式，对长词进行分词 cut_all: False
def dm03():
    text = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能。"
    result = jieba.cut_for_search(text)
    print(result)
    result = jieba.lcut_for_search(text)
    print(result)


# 繁体分词： jieba繁体分词和简体是一样的
def dm04():
    text = "煩惱即是菩提，我暫且不提"
    result = jieba.lcut(text)
    print('繁体模式', result)


# 用户自定义词典
def dm05():
    text = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能。"
    result = jieba.lcut(text)
    print(result)

    jieba.load_userdict("userdict.txt")
    result = jieba.lcut(text)
    print(result)


# 词性分组
def dm06():
    import jieba.posseg as pseg
    text = "我爱杭州西湖，还喜欢灵隐寺。"
    result = pseg.lcut(text)
    print(result)
    for word, flag in result:
        print(word, flag)


# print(shape([[[1,2,3],[4,5,6]]]))
print(torch.cuda.is_available())
print(torch.version.cuda)

if __name__ == '__main__':
    print("==================jieba分词====================")
    print("==================精准模式==================", '\n')
    dm01()
    print("==================全模式==================", '\n')
    dm02()
    print("==================搜索引擎模式==================", '\n')
    dm03()
    print("==================繁体分词==================", '\n')
    dm04()
    print("==================用户自定义词典==================", '\n')
    dm05()
    print("==================词性分组==================")
    dm06()
