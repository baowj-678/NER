""" 数据集加载器
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/12/16
"""
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
from vocab import Vocab
from dataset import DataSet

def collate_func(X):
    """ batch数据处理 (word_list, entity_list, length)
    @param:
    :X (word_list, entity_list, length) 从__getitem__返回的list
    @return:
    :word packed_sequence
    :entity packed_sequence
    """
    word_lists = []
    entity_lists = []
    length_list = []
    for i in X:
        word_lists.append(i[0])
        entity_lists.append(i[1])
        length_list.append(i[2])
    word_lists = np.array(word_lists)
    entity_lists = np.array(entity_lists)
    length_list = np.array(length_list)
    # Sort
    x_indices = np.argsort(length_list)[::-1]
    length_list = length_list[x_indices]
    word_lists = word_lists[x_indices]
    entity_lists = entity_lists[x_indices]
    # Pack
    # word = pack_padded_sequence(word_lists, length_list, batch_first=True)
    # entity = pack_padded_sequence(entity_lists, length_list, batch_first=True)
    return (word_lists, entity_lists, length_list)


class DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=16,shuffle=True):
        super(DataLoader, self).__init__(dataset=dataset, # 数据集
                                         batch_size=batch_size, # batch_size
                                         shuffle=shuffle, # 打乱
                                         sampler=None,
                                         batch_sampler=None,
                                         num_workers=0,
                                         collate_fn=collate_func,
                                         pin_memory=False,
                                         drop_last=False,
                                         timeout=0,
                                         worker_init_fn=None,
                                         multiprocessing_context=None)


if __name__ == '__main__':
    vocab = Vocab('D:/NLP/NER/dataset/CONLL2003/vocab.txt')
    labels = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG']
    entity2index = {}
    index2entity = {}
    for i, label in enumerate(labels):
        entity2index[label] = i
        index2entity[i] = label
    dataset = DataSet(path='D:/NLP/NER/dataset/CONLL2003/test.txt',
                      vocab=vocab,
                      entity2index=entity2index)
    dataloader = DataLoader(dataset)
    for epoch, data in enumerate(dataloader):
        break