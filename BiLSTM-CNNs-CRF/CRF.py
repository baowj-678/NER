""" BiLSTM-CNN-CRF的CRF部分
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/11/1
"""
from typing import Optional
import torch
import torch.nn as NN
import torch.distributions as tdist
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import copy


class CRF(NN.Module):
    def __init__(self, hidden_size, class_num, device, entity_vocab, k=3):
        super(CRF, self).__init__()
        """ 初始化\n
        @param:\n
        :hidden_size:\n
        :class_num:\n
        :device:\n
        """
        # Device
        self.device = device
        # 集束搜索的K
        self.k = k
        self._entity_start_ = entity_vocab.start_i
        self._entity_end_ = entity_vocab.end_i
        # 参数初始化(mu=0, sigma=1.0)
        self.W = NN.Parameter(torch.zeros((class_num, class_num, 1, hidden_size), dtype=torch.float64, requires_grad=True).to(device))
        self.b = NN.Parameter(torch.zeros((class_num, class_num), dtype=torch.float64, requires_grad=True).to(device))
        NN.init.uniform_(self.W, 0, 1)
        NN.init.uniform_(self.b, 0, 1)
        # Tanh
#######################################################################
        self.softmax = NN.Softmax(dim=-1)
#######################################################################


    def forward(self, input, entity=None, length: Optional[int]=None):
        """ 前向传播\n
        @param:\n
        :input: (batch_size, seq_length, hidden_size) 输入的数据\n
        :length: (batch_size) 句子长度\n
        :entity: (batch_size, seq_length) 正确标签\n
        @return:
        :Loss: (tensor) 概率的熵值\n
        """
        is_packed = False
        if length is None:
            input, length = pad_packed_sequence(input)
            is_packed = True
        seq_length, batch_size, hidden_size = input.shape
        input = input.reshape(batch_size, seq_length, 1, 1, hidden_size, 1) 
        # (batch_size, seq_length, 1, 1, hidden_size, 1)
        P = torch.matmul(self.W, input)
        # (batch_size, seq_length, class_num, class_num, 1, 1)
        P = P.squeeze(dim=-1).squeeze(dim=-1)
        # (batch_size, seq_length, class_num, class_num)
        P = P.permute([1, 0, 2, 3])
        # (seq_length, batch_size, class_num, class_num)
        P = P + self.b
##################################################################
        P = self.softmax(P)
##################################################################
        # (seq_length, batch_size, class_num, class_num)
        if entity is None:
            return self.predict(P, length)
        else:
            P_UP = self.p_up(P, length, entity)
            P_DOWN = self.p_down(P, length)
            # (batch_size)
            P_FINAL = P_DOWN - P_UP
            Loss = torch.sum(P_FINAL)
            return Loss

    def p_down(self, P, length):
        """ 计算概率和\n
        @param:\n
        :P: (seq_length, batch_size, class_num, class_num) Wz+b\n
        :length: (batch_size)\n
        @return:\n
        :p_down (batch_size) log(sigma(prob(exp(Wz+b))))\n
        """
        # P = torch.exp(P)
        p_down = P[0, :, self._entity_start_, :].unsqueeze(dim=1) 
        # (batch_size, 1, class_num) t=0步，第一个单词为START情况下，下个单词为i的概率
        p_sums = [p_down]
        for P_t in P[1:]: 
            # (batch_size, class_num, class_num)
            p_down = torch.matmul(p_down, P_t) # (batch_size, 1, class_num)
            p_sums.append(p_down)
        p_final = []
        for i, l in enumerate(length):
            p_final.append(p_sums[l - 1][i, :, :]) # (1, 1, class_num)
        p_final = torch.cat(p_final, dim=0) # (batch_size, class_num)
        p_ans = torch.sum(p_final, dim=1) 
        # (batch_size)
        p_down = torch.log(p_ans)
        # (batch_size)
        return p_down
    
    def p_up(self, P, length, entity):
        """ 计算输入序列的概率\n
        @param:\n
        :P: (seq_length, batch_size, class_num, class_num) 各个Wz+b的结果\n
        :length: (batch_size) 每个batch的句长\n
        :entity: [list](bacth_size, seq_length) 每个单词的entity\n
        @return:\n
        :p_up: (batch_size) 每句话的Wz+b\n
        """
        P = P.permute([1, 0, 2, 3])
        # (batch_size, seq_length, class_num, class_num)
        P_UP = []
        for i, P_i in enumerate(P):
            # (seq_length, class_num, class_num)
            p_tmp = 0
            last_entity = self._entity_start_
            for t in range(length[i]):
                # (class_num, class_num)
                P_i_t = P_i[t]
                next_entity = entity[i][t]
############################################################################################
                p_tmp += torch.log(P_i_t[last_entity, next_entity])
############################################################################################                last_entity = next_entity
            P_UP.append(p_tmp)
        P_UP = torch.tensor(P_UP, device=self.device)
        return P_UP

    def predict(self, P, length):
        """ P各个时间步的转移概率
        @param:
        :P: (seq_length, batch_size, class_num, class_num)\n
        :length: (batch_size) 每个batch的句长\n
        @return:\n
        :batch_label_list [list]
        """
        P = torch.exp(P)
        P = P.permute([1, 0, 2, 3])
        # (batch_size, seq_length, class_num, class_num)
        batch_label_list = []
        for k, p_batch in enumerate(P):
            # (seq_length, class_num, class_num)
            p_t = p_batch[0, self._entity_start_]
            value_t, label_t = torch.topk(p_t, self.k)
            label_list = [[label_t[i]] for i in range(self.k)]
            value_list = value_t
            for p_t in p_batch[1:length[k]]: 
                # (class_num, class_num)
                label_list_ = []
                value_list_ = []
                for index, label_value in enumerate(zip(label_t, value_t)):
                    # 分别对各个label进行搜索
                    label_t_i, value_t_i = label_value
                    value_t_, label_t_ = torch.topk(p_t[label_t_i], self.k)
                    value_t_ = value_t_ * value_t_i
                    label_list_.append(label_t_)
                    value_list_.append(value_t_)
                # label_list_ = torch.cat(label_list_) # (k*k)
                value_list_ = torch.cat(value_list_) # (k*k)
                value_list_, topk_label = torch.topk(value_list_, self.k)
                value_t = value_list_
                label_list_new = []
                for i, topk_label_i in enumerate(topk_label): # topk的label
                    label_t[i] = label_list_[topk_label_i // self.k][topk_label_i % self.k]
                    label_list_new.append(copy.deepcopy(label_list[topk_label_i // self.k]) + [label_t[i]])
                    # 原label_list + new_label
                label_list = label_list_new
            value, indice = torch.max(value_list, dim=0)
            batch_label_list.append(torch.stack(label_list[indice]).cpu().numpy())
        return batch_label_list


if __name__ == "__main__":
    batch_size = 3
    seq_length = 10
    class_num = 5
    hidden_size = 12
    P = torch.ones(size=(seq_length, batch_size, class_num, class_num))
    P[5, 0, 2, 3] = 2
    # P = torch.randn(size=(seq_length, batch_size, class_num, class_num))
    length = torch.tensor([10, 10, 10])
    crf = CRF(hidden_size=hidden_size, class_num=class_num, device=torch.device('cpu'), entity_start=0, k=5)
    print(crf.predict(P))