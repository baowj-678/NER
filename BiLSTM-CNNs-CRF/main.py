""" BiLSTM-CNN-CRF的main函数部分
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/12/16
"""
import pandas as pd
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import torch
import sys
import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
from dataset import Vocab
from dataset import DataSet
from dataset import DataLoader
from Model import Model
from seed_all import seed_all


if __name__ == '__main__':
    # 设置随机种子
    seed_all(42)
    # 设置路径
    train_data_path = 'D:/NLP/NER/dataset/CONLL2003/test.txt'
    dev_data_path = 'D:/NLP/NER/dataset/CONLL2003/valid.txt'
    test_data_path = 'D:/NLP/NER/dataset/CONLL2003/test.txt'
    vocab_path = 'D:/NLP/NER/dataset/CONLL2003/vocab.txt'
    save_path = 'D:/NLP/NER/BiLSTM-CNNs-CRF/output/BiLSTM-CNN-CRF.pkl'
    glove_file=None,
    word2vec_file='D:/NLP/Word2Vec/GloVe/glove6B50d.txt',

    BATCH_SIZE = 32
    max_length = 56
    hidden_size = 128
    char_embedding_dim=128 # charater-level embedding dim
    word_embedding_dim=50  # word-level embedding dim
    max_sent_length = 20
    lr = 3e-3
    output_per_batchs = 10
    test_per_batchs = 60
    # 创建entity -> index映射
    entitys = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG']
    entity2index = {}
    index2entity = {}
    for i, label in enumerate(entitys):
        entity2index[label] = i
        index2entity[i] = label
    # 加载字典
    vocab = Vocab(vocab_path)
    # 创建数据集
    train_data_set = DataSet(path=train_data_path,
                        vocab=vocab,
                        entity2index=entity2index)
    test_data_set = DataSet(path=test_data_path,
                        vocab=vocab,
                        entity2index=entity2index)
    # 创建加载器
    train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data_set, shuffle=True, batch_size=BATCH_SIZE)
    # 是否用GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # 模型初始化
    model = Model(vocab=vocab,
                  char_embedding_dim=char_embedding_dim,
                  word_embedding_dim=word_embedding_dim,
                  hidden_size=hidden_size,
                  device=device,
                  kernel_n=5, # 卷积核长度
                  padding=2,  # padding大小
                  word2vec_file=word2vec_file,
                  max_sent_length=max_sent_length,
                  class_num=len(entity2index),
                  dropout=0.1)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-3)
    # 开始训练
    for i in range(100):
        print('='*8 + '开始训练' + '='*8)
        model.train()
        loss_sum = 0
        for epoch, data in enumerate(train_data_loader):
            X, Y = data
            optimizer.zero_grad()
            loss = model(X, Y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            # 打印训练情况
            if((epoch + 1) % output_per_batchs == 0):
                print('itor: {}: epoch: {}  loss: {}'.format(i + 1, epoch + 1, loss_sum / output_per_batchs))
                loss_sum = 0
            ############################### 测试 ######################################
            if (epoch + 1) % test_per_batchs == 0:
                print('-'*8 + '开始测试' + '-'*8)
                with torch.no_grad():
                    accuracy = 0
                    model.eval()
                    for epoch, data in enumerate(test_data_loader):
                        X, Y = data
                        Y = Y.to(device=device).squeeze(dim=1)
                        y = model(X).detach()
                        accuracy += torch.sum(y == Y).cpu()
                    print('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, len(test_data_set), float(accuracy) / len(test_data_set)))
                    torch.save(model.state_dict(), save_path)
                model.train()
