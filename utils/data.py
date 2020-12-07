# -*- coding:utf-8 -*-
""" -------------read data------------- """


def read_data(path):
    """
    读取原始数据，不需要进行预处理，因为需要读取的文件格式不同，但预处理的操作类似，解耦合
    :param path: 数据的路径,txt, csv
    :return: 返回的格式，string list
    """
    text_list = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.read().splitlines():
            if line:
                text_list.append(line.strip()[:100])
    return text_list[:10]


""" -------------process data------------- """


def clean(text_list, stop_path):
    """
    数据清洗，删除停用词，标点，链接，特殊字符串，
    :param text_list: string list [string]
    :param stop_path: 停用词路径
    :return: 处理后的句子合并成一篇文章
    """
    lines = []
    stopwords = [line.strip() for line in open(stop_path, 'r').readlines()]
    for line in text_list:
        stop_free = " ".join([i for i in line.split() if i not in stopwords])
        lines.append(stop_free)
    # 中文不需要空格
    doc = ''.join(line for line in lines)
    return doc


class Vocab:
    def __init__(self, doc):
        self.build_corpus(doc)
    """ -------------build corpus------------- """
    def build_corpus(self, doc):
        """
        进行分词处理，构建vocab，word2idx ....
        :param doc: 常字符串，代表所有训练数据拼接的结果
        :return:
        """
        vocab = set(doc)
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
