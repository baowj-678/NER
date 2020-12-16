""" 字典的文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/30
"""
import os
from typing import Optional
import nltk


class Vocab:
    def __init__(self, vocab_path):
        """ 字典初始化
        @param:
        :vocab_path: 字典文件路径
        """
        super().__init__()
        self._stoi_ = {}
        self._itos_ = {}
        self._pad_s_ = '_pad_'
        self._pad_i_ = 0
        self._stoi_['_pad_'] = self._pad_i_
        self._itos_[self._pad_i_] = self._pad_s_
        print('-'*8, 'load vocab', '-'*8)
        self.load_file(path=vocab_path)
        print('-'*8, 'load successfully', '-'*8)
        self.max_l = 0

    def load_file(self, path):
        """ 加载字典文件 """
        if not os.path.exists(path):
            raise Exception("文件不存在")
        with open(path, mode='r') as file:
            data = file.readlines()
        begin = 1
        for word in data:
            word = word.strip()
            if word not in self._stoi_:
                self._stoi_[word] = begin
                self._itos_[begin] = word
                begin += 1
        return self._stoi_
    
    def wordlist2index(self, word_list, max_length: Optional[str]):
        """ 将一组单词转成index
        @param:\n
        :word_list [list][str]: 切成单词的句子\n
        :max_length [int]: 句子的最大长度(padding & cutting)
        @return:\n
        :index_list [list][int]: 单词对应的index列表(padding & cutting)\n
        :length_list [int]: 原句子长度(=None,未padding & cutting)\n
        """
        index_list = None
        index_length = None
        if max_length is None:
            # not padding
            index_list = [self.__getitem__(word) for word in word_list]
        else:
            # padding
            if len(word_list) > max_length:
                # cutting
                index_list = [self.__getitem__(word) for word in word_list[:max_length]]
                index_length = max_length
            else:
                # padding
                index_list = [self.__getitem__(word) for word in word_list] + [self._pad_i_] * (max_length - len(word_list))
                index_length = len(word_list)
        return (index_list, index_length)

    def wordlists2index(self, word_lists, max_length: Optional[str]):
        """ 将一组单词转成index
        @param:\n
        :word_lists [list](batch_size, sent_length)[str]: 切成单词的句子\n
        :max_length [int]: 句子的最大长度(padding & cutting)
        @return:\n
        :index_list [list][int]: 单词对应的index列表(padding & cutting)\n
        :length_list [int]: 原句子长度(=None,未padding & cutting)\n
        """
        index_list = []
        length_list = []
        for word_list in word_lists:
            index, length = self.wordlist2index(word_list, max_length)
            index_list.append(index)
            length_list.append(length)
        return (index_list, length_list)

    def sents2indexs(self, sents, max_length=None):
        """ 句子转index
        @param:
        :sents: (list) 句子s
        :max_length: (int) 最大句长
        @return:
        :indexs: (list) padding好的句子s
        :lengths: (list) 原句子s长度
        """
        indexs = []
        lengths = []
        for sent in sents:
            index, length = self.sent2index(sent, max_length)
            indexs.append(index)
            lengths.append(length)
        return (indexs, lengths)

    def sent2index(self, sent: str, max_length=None):
        """ 单个句子(str)转index(max_length)
        @param:
        :sent: (str)句子
        :max_length: (int) 最大句长
        @return:
        :sent: (list) padding好的indexs
        :length: int 原句长
        """
        sent = nltk.word_tokenize(sent)
        return self.wordlist2index(sent)

    def __len__(self):
        return len(self._stoi_)

    def __getitem__(self, word):
        """ 根据word查询index """
        return self._stoi_.get(word, self._pad_i_)

    def word2index(self, word):
        """ 根据word查询index """
        return self._stoi_.get(word, self._pad_i_)

    @property
    def pad_i(self):
        """ 返回pad index """
        return self._pad_i_

    @property
    def pad_s(self):
        """ 返回pad str"""
        return self._pad_s_

    def itos(self, index):
        """ 根据index查询word """
        return self._itos_.get(index, None)
    
    def index2word(self, index):
        """ 根据index查询word """
        return self._itos_.get(index, None)

if __name__ == '__main__':
    vocab = Vocab('D:/NLP/TC/dataset/SST-1/vocab.txt')
    print(len(vocab))
    print(vocab.sent2index('Yet the act is still charming here .'))
    print(vocab.pad_s)