""" 数据集类(包括数据加载)
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/12/16
"""
import sys
import os
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
from torch.utils.data import Dataset
from vocab import Vocab, EntityVocab
from utils import *


class DataSet(Dataset):
    def __init__(self, path, vocab, entity_vocab, entity_padding_len):
        """ 初始化\n
        @param:\n
        :path: 文件路径\n
        :vocab: 字典\n
        :entity2index: entity->index\n
        """
        super(DataSet, self).__init__()
        self.vocab = vocab
        self.word_lists = None
        self.entity_lists = None
        self.length_list = None
        self.entity_padding_len = entity_padding_len
        self.entity_vocab = entity_vocab
        self.load_data(path, delimiter=' ', entity_vocab=entity_vocab)

    def __getitem__(self, index):
        return (self.word_lists[index],     # 句子
                self.entity_lists[index],   # 实体标签
                self.length_list[index])    # 句子长度

    def __len__(self):
        return len(self.word_lists)

    def load_data(self, path, entity_vocab, delimiter=" ", word_index=0, entity_index=3):
        """ 加载数据\n
        @param:\n
        :path: (str) 文件路径\n
        :vocab: (Vocab) 字典\n
        :entity2index: (dict) entity->index\n
        :delimier: 分隔符\n
        :word_index: (int) 单词的索引\n
        :entity_index: (int) 实体的索引\n
        @return:
        :word_lists: [list](batch_size, sent_len) 单词(包括大写)\n
        :entity_lists: [list](batch_size, sent_len) 对应的实体索引\n
        """
        self.word_lists = []
        self.entity_lists = []
        self.length_list = []
        word_list = []
        entity_list = []
        """ 单文件处理 """
        data = DataDeal.load_data(path=path, delimiter=delimiter)
        for line in data:
            if len(line) > 1:
                word = line[word_index]
                if word == '-DOCSTART-':
                    continue
                word_list.append(word)
                entity = line[entity_index]
                entity_list.append(entity_vocab[entity])
            else:
                if len(word_list) > 0:
                    self.word_lists.append(word_list)
                    self.entity_lists.append(entity_list)
                    self.length_list.append(len(word_list))
                    word_list = list()
                    entity_list = list()
        if len(word_list)> 0:
                self.word_lists.append(word_list)
                self.entity_lists.append(entity_list)
                self.length_list.append(len(word_list))
        return (self.word_lists, self.entity_lists)


if __name__ == '__main__':
    vocab = Vocab('D:/NLP/NER/dataset/CONLL2003/vocab.txt')
    entity_vocab = EntityVocab('D:/NLP/NER/dataset/CONLL2003/entity.txt')
    dataset = DataSet(path='D:/NLP/NER/dataset/CONLL2003/test.txt',
                      vocab=vocab,
                      entity_vocab=entity_vocab,
                      entity_padding_len=10)
    print(dataset[9])