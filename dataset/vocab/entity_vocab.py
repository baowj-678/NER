""" 实体典的文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/12/17
@github: https://github.com/baowj-678
"""
import os
from typing import Optional

class EntityVocab:
    def __init__(self, vocab_path):
        """ 字典初始化
        @param:
        :vocab_path: 字典文件路径
        """
        super().__init__()
        self._stoi_ = {}
        self._itos_ = {}
        # 辅助字符
        self._start_s_ = '_start_'
        self._end_s_ = '_end_'
        self._start_i_ = 0
        self._end_i_ = 1
        # 正规字符开始标签
        self.begin = 2
        self._stoi_[self._end_s_] = self._end_i_
        self._stoi_[self._start_s_] = self._start_i_
        self._itos_[self._end_i_] = self._end_s_
        self._itos_[self._start_i_] = self._start_s_
        print('-'*8, 'load entity-vocab', '-'*8)
        self.load_file(path=vocab_path)
        print('-'*8, 'load successfully', '-'*8)

    def load_file(self, path):
        """ 加载字典文件 """
        if not os.path.exists(path):
            raise Exception("文件不存在")
        with open(path, mode='r') as file:
            data = file.readlines()
        begin = self.begin
        for entity in data:
            entity = entity.strip()
            if entity not in self._stoi_:
                self._stoi_[entity] = begin
                self._itos_[begin] = entity
                begin += 1
        return self._stoi_
    
    def sent2index(self, sent, padding_length: Optional[int]=None):
        """ 给一个单词编码
        @param:\n
        :sent [list] 实体句子\n
        :padding_length [int] padding后的长度\n
        """
        if padding_length is None:
            padding_length = len(sent)
        indexs = [self._end_i_] * padding_length
        if padding_length < len(sent):
            for i, c in enumerate(sent[:padding_length]):
                indexs[i] = self._stoi_[c]
        else:
            for i, c in enumerate(sent):
                indexs[i] = self._stoi_[c]
        return indexs
    
    def sents2index(self, sents, padding_length: Optional[int]=None):
        """ 给一组实体编号
        @param:\n
        :sents [list] 一些实体句子\n
        :padding_length [int] padding后的长度\n
        @return:\n
        :indexs [list](batch_size, seq_length)\n
        """
        indexs = []
        for sent in sents:
            indexs.append(self.sent2index(sent, padding_length))
        return indexs

    def __len__(self):
        return len(self._stoi_)

    def __getitem__(self, word):
        """ 根据word查询index """
        return self._stoi_.get(word, -1)

    def entity2index(self, word):
        """ 根据word查询index """
        return self._stoi_.get(word, -1)

    @property
    def end_i(self):
        """ 返回end index """
        return self._end_i_

    @property
    def start_i(self):
        """ 返回start index """
        return self._start_i_

    @property
    def start_s(self):
        """ 返回start str"""
        return self._start_s_

    @property
    def end_s(self):
        """ 返回end str"""
        return self._end_s_

    def itos(self, index):
        """ 根据index查询word """
        return self._itos_.get(index, None)
    
    def index2entity(self, index):
        """ 根据index查询word """
        return self._itos_.get(index, None)