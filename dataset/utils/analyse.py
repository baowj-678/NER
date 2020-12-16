""" 数据集分析
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/31
"""
import os
import sys
import matplotlib.pyplot as plt
import nltk
import json
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
from data_deal import DataDeal



class Analyse:
    def __init__(self, path):
        """ 初始化
        @param:
        :path: (str) 文件路径
        """

    @staticmethod
    def analyse(path,
                delimiter=' ',
                word_index=1,
                entity_index=-1,
                has_head=True,
                length_analyse=True,
                word_analyse=True,
                label_analyse=True,
                save_path=None):
        """ 生成字典文件
        :path: (list/str) 原文件
        :tgt_path: 目标文件存放位置
        :head: 句子所在的列
        :delimiter: 分割符
        """
        vocab = dict()
        entities = dict()
        max_length = 0
        sum_length = 0
        sum_lines = 0
        lengths = dict()
        print('-' * 8, 'analyse', '-' * 8)
        def make_vocab_single(path):
            """ 单文件处理 """
            data = DataDeal.load_data(path=path, delimiter=delimiter)
            seq_length = 0
            for line in data:
                if len(line) > 1:
                    seq_length += 1
                    word = line[word_index]
                    word = word.lower()
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                    entity = line[entity_index]
                    if entity in entities:
                        entities[entity] += 1
                    else:
                        entities[entity] = 1
                else:
                    # sum_lines += 1


                    
                    if seq_length in lengths:
                        lengths[seq_length] += 1
                    else:
                        lengths[seq_length] = 1
                    seq_length = 0
                
        if isinstance(path, list):
            for _ in path:
                make_vocab_single(_)
        else:
            make_vocab_single(path)
        print("-" * 8, "result", '-' * 8)
        font = {'family': 'SimHei',
                'style': 'italic',
                'weight': 'normal',
                'color': 'black',
                'size': 20
                }
        if length_analyse:
            print('max sentence length:{}'.format(max_length))
            print('average sentence length:{}'.format(sum_length/sum_lines))
            plt.scatter(list(lengths.keys()), list(lengths.values()))
            plt.title('sentence length distribution', font)
            plt.ylabel('length', font)
            plt.xlabel('nums', font)
            plt.grid()
            plt.show()
        if word_analyse:
            word_time = dict()
            for time in vocab.values():
                if time in word_time:
                    word_time[time] += 1
                else:
                    word_time[time] = 1
            plt.scatter(list(word_time.keys()), list(word_time.values()))
            plt.title('word show times distribution', font)
            plt.ylabel('nums', font)
            plt.xlabel('show-times', font)
            plt.grid()
            plt.show()
            DataDeal.save_dict_json(path=save_path, dict_=vocab)
        return

    @staticmethod
    def word_analyse():
        pass


if __name__ == '__main__':
    Analyse.analyse(path=['dataset/CONLL2003/train.txt', 'dataset/CONLL2003/test.txt', 'dataset/CONLL2003/valid.txt'],
                    save_path='dataset/CONLL2003/vocab_analyse.json')