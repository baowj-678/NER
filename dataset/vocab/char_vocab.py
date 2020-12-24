""" 字母典的文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/12/17
@github: https://github.com/baowj-678
"""
import os
from typing import Optional

class CharVocab:
    def __init__(self, vocab_path):
        """ 字典初始化
        @param:
        :vocab_path: 字典文件路径
        """
        super().__init__()
        self._stoi_ = {}
        self._itos_ = {}
        # 辅助字符
        self._pad_s_ = '_pad_'
        self._pad_i_ = 0
        # 正规字符开始标签
        self.begin = 1
        self._stoi_['_pad_'] = self._pad_i_
        self._itos_[self._pad_i_] = self._pad_s_
        print('-'*8, 'load char-vocab', '-'*8)
        self.load_file(path=vocab_path)
        print('-'*8, 'load successfully', '-'*8)
        self.max_l = 0

    def load_file(self, path):
        """ 加载字典文件 """
        if not os.path.exists(path):
            raise Exception("文件不存在")
        with open(path, mode='r') as file:
            data = file.readlines()
        begin = self.begin
        for word in data:
            word = word.strip()
            if word not in self._stoi_:
                self._stoi_[word] = begin
                self._itos_[begin] = word
                begin += 1
        return self._stoi_
    
    def word2index(self, word: str, padding_length: Optional[int]=None):
        """ 给一个单词编码
        @param:\n
        :word [str] 单词\n
        :padding_length [int] padding后的长度\n
        """
        if padding_length is None:
            padding_length = len(word)
        indexs = [self._pad_i_] * padding_length
        for i, c in enumerate(word):
            indexs[i] = self._stoi_(c)
        return indexs

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