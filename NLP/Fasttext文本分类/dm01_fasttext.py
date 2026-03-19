import fasttext


# 模型训练
def dm01():
    model = fasttext.train_supervised('./data/cooking_train.txt')
    print('model-->', model)
    model.save_model('./cooking_train.bin')


def dm02():
    # 加载模型
    model = fasttext.load_model('./cooking_train.bin')
    # 预测
    result = model.predict('How do I make a chocolate cake?')
    print('result-->', result)
    result = model.test('./data/cooking_valid.txt')
    print('result1-->', result)
    # (3000, 0.13833333333333334, 0.0598241314689347)


def dm03():
    model = fasttext.train_supervised('./data/cooking.pre.train')
    model.save_model('./cooking.pre.bin')


def dm04():
    model = fasttext.load_model('./cooking.pre.bin')
    result = model.predict('How do I make a chocolate cake?')
    print('result-->', result)
    result = model.test('./data/cooking.pre.valid')
    print('result1-->', result)
    # (3000, 0.17233333333333334, 0.07452789390226322)


# 增加训练轮次和学习率
def dm05():
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=25, lr=1.0)
    # model.save_model('./cooking.pre.bin')

    result = model.predict('How do I make a chocolate cake?')
    print('result-->', result)
    result = model.test('./data/cooking_valid.txt')
    print('result1-->', result)
    # (3000, 0.47633333333333333, 0.20599682860025947)


# 增加n_gram特征
def dm06():
    # model = fasttext.train_supervised('./data/cooking.pre.train', epoch=25, lr=1.0, wordNgrams=2)
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=25, lr=1.0, wordNgrams=2,loss='ova')

    result = model.predict('How do I make a chocolate cake?')
    print('result-->', result)
    result = model.test('./data/cooking_valid.txt')
    print('result1-->', result)
    # (3000, 0.5013333333333333, 0.21680841862476574)


if __name__ == '__main__':
    # dm01()
    dm02()
    # dm03()
    dm04()
    dm05()
    dm06()
