import fasttext


# 训练词向量模型
def dm01():
    model = fasttext.train_unsupervised('./data/fil9', epoch=1, dim=100)
    print('model-->', model)
    model.save_model('./data/fil9.bin')


def dm02():
    # 加载模型
    model = fasttext.load_model('./data/fil9.bin')
    # 获取token的词向量
    result = model.get_word_vector('dog')
    print('result-->', result)
    result = model.get_nearest_neighbors('dog')
    print('result-->', result)


if __name__ == '__main__':
    # dm01()
    dm02()
