""" 根据原文件生成字典文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/30
"""
import nltk
import os
import sys
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
from dataset.utils import *


class MakeVocab():
    def __init__(self):
        pass

    @staticmethod
    def make_vocab(src_path, word_index, entity_index, save_path):
        """ 加载Conll2003数据集
        @param:
        :src_path: (list/str) 文件路径
        @return:
        :data list(sent, name-entity): 语句和命名实体
        """
        vocab = {}
        entitys = {}
        def make_vocab_single(path):
            """ 单文件处理 """
            data = DataDeal.load_data(path=path, delimiter=' ')
            for line in data:
                if len(line) > 1:
                    word = line[word_index]
                    word = word.lower()
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                    entity = line[entity_index]
                    if entity in entitys:
                        entitys[entity] += 1
                    else:
                        entitys[entity] = 1

        if isinstance(src_path, list):
            for path in src_path:
                make_vocab_single(path)
        else:
            make_vocab_single(src_path)
        vocab_save_path = os.path.join(save_path, 'vocab.txt')
        vocab_analyse_save_path = os.path.join(save_path, 'vocab_analyse.json')
        entity_save_path = os.path.join(save_path, 'entity.txt')
        entity_analyse_save_path = os.path.join(save_path, 'entity_analyse.json')
        DataDeal.save_dict_json(vocab_analyse_save_path, vocab)
        DataDeal.save_dict_json(entity_analyse_save_path, entitys)
        DataDeal.save_single(vocab_save_path, list(vocab.keys()))
        DataDeal.save_single(entity_save_path, list(entitys.keys()))
        return

if __name__ == '__main__':
    # MakeVocab.make_vocab_csv(['D:/NLP/TC/dataset/SST-1/train.tsv', 'D:/NLP/TC/dataset/SST-1/test.tsv', 'D:/NLP/TC/dataset/SST-1/dev.tsv'],
    #                          tgt_path='D:/NLP/TC/dataset/SST-1/vocab.txt',
    #                          seq_index=1,
    #                          delimiter='\t')
    pass
    # print(nltk.word_tokenize("The Rock is destined to be the 21st Century s newConan '' and that he s going to make a splash even greater than; it's Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal"))