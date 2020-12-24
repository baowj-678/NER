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
from dataset import Vocab, CharVocab, EntityVocab
from dataset import DataSet
from dataset import DataLoader
from Model import Model
from seed_all import seed_all


if __name__ == '__main__':
    # 设置随机种子
    seed_all(42)
    # 设置路径
    data_dir = 'D:/NLP/NER/dataset/CONLL2003/'
    train_data_path = os.path.join(data_dir, 'train.txt')
    dev_data_path = os.path.join(data_dir, 'valid.txt')
    test_data_path = os.path.join(data_dir, 'test.txt')
    vocab_path = os.path.join(data_dir, 'vocab.txt')
    entity_vocab_path = os.path.join(data_dir, 'entity.txt')
    char_vocab_path = os.path.join(data_dir, 'char_vocab.txt')
    save_path = 'D:/NLP/NER/BiLSTM-CNNs-CRF/output/BiLSTM-CNN-CRF.pkl'
    glove_file=None,
    word2vec_file='D:/NLP/Word2Vec/GloVe/love6B50d.txt',

    BATCH_SIZE = 16
    hidden_size = 128
    char_embedding_dim=128 # charater-level embedding dim
    word_embedding_dim=50  # word-level embedding dim
    max_sent_length=35
    max_word_length=16
    kernel_n=3 # 卷积核长度
    padding=2 # padding大小
    lr = 3e-3
    weight_decay = 1e-3 # 梯度衰减权值
    gradient_clipping = 5 # 梯度裁剪
    output_per_batchs = 1
    test_per_batchs = 5
    test_batchs = 1
    ITORS = 100
    # 加载字典
    vocab = Vocab(vocab_path)
    char_vocab = CharVocab(char_vocab_path)
    entity_vocab = EntityVocab(entity_vocab_path)
    # 创建数据集
    train_data_set = DataSet(path=train_data_path,
                        vocab=vocab,
                        entity_vocab=entity_vocab,
                        entity_padding_len=max_sent_length)
    test_data_set = DataSet(path=test_data_path,
                        vocab=vocab,
                        entity_vocab=entity_vocab,
                        entity_padding_len=max_sent_length)
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
                  char_vocab=char_vocab,
                  entity_vocab=entity_vocab,
                  char_embedding_dim=char_embedding_dim,
                  word_embedding_dim=word_embedding_dim,
                  hidden_size=hidden_size,
                  device=device,
                  kernel_n=kernel_n, # 卷积核长度
                  padding=padding,  # padding大小
                  word2vec_file='D:/NLP/Word2Vec/GloVe/glove6B50d.txt',
                  max_sent_length=max_sent_length,
                  max_word_length=max_word_length,
                  class_num=len(entity_vocab),
                  dropout=0.1)
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    # 开始训练
    for i in range(ITORS):
        print('='*8 + '开始训练' + '='*8)
        model.train()
        loss_sum = 0
        for epoch, data in enumerate(train_data_loader):
            word_lists, entity_lists, length_list = data
            optimizer.zero_grad()
            loss = model(word_lists, entity_lists)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping, norm_type=2)
            optimizer.step()
            loss_sum += loss.detach()
            # 打印训练情况
            if((epoch + 1) % output_per_batchs == 0):
                print('itor: {}/{}: epoch: {}/{}  loss: {}'.format(i + 1, ITORS, epoch + 1, len(train_data_loader), loss_sum / output_per_batchs))
                loss_sum = 0
            ############################### 测试 ######################################
            if (epoch + 1) % test_per_batchs == 0:
                print('-'*8 + '开始测试' + '-'*8)
                with torch.no_grad():
                    accuracy = 0
                    model.eval()
                    total = 0
                    for epoch, data in enumerate(test_data_loader):
                        word_lists, entity_lists, length_list = data
                        # print(entity_lists)
                        predict_entity = model(word_lists)
                        # print(entity_lists)
                        accuracy = 0
                        total = 0
                        for target, predict in zip(entity_lists, predict_entity):
                            total += len(target)
                            accuracy += np.sum(np.array(target) == predict)
                        if (epoch + 1) % test_batchs == 0:
                            break
                    print('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, total, float(accuracy) / len(test_data_set)))
                    torch.save(model.state_dict(), save_path)
                model.train()
