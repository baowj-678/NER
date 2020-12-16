""" BiLSTM-CNN-CRF的CRF部分
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/11/1
"""
import torch
import torch.nn as NN
import torch.distributions as tdist
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence 
import copy


class CRF(NN.Module):
    def __init__(self, hidden_size, class_num, device, label_start=0, k=3):
        super(CRF, self).__init__()
        """ 初始化\n
        @param:\n
        :hidden_size:\n
        :class_num:\n
        :device:\n
        """
        self.k = k
        self._label_start_ = label_start
        # 参数初始化(mu=0, sigma=1.0)
        Sampler = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        # Parameters
        self.W = NN.Parameter(Sampler.sample(sample_shape=(class_num, class_num, 1, hidden_size)), requires_grad=True).to(device)
        self.b = NN.Parameter(Sampler.sample(sample_shape=(class_num, class_num)), requires_grad=True).to(device)
    
    def forward(self, input, length=None, label=None):
        """ 前向传播\n
        @param:\n
        :input: (batch_size, seq_length, hidden_size) 输入的数据\n
        :length: (batch_size) 句子长度\n
        :label: (batch_size, seq_length) 正确标签\n
        @return:
        :
        """
        is_packed = False
        if length is None:
            input, length = pad_packed_sequence(input)
            is_packed = True
        batch_size, seq_length, hidden_size = input.shape
        input = input.reshape(batch_size, seq_length, 1, 1, hidden_size, 1) 
        # (batch_size, seq_length, 1, 1, hidden_size, 1)
        P = torch.matmul(self.W, input)
        P = P.squeeze(dim=-1).squeeze(dim=-1)
        # (batch_size, seq_length, class_num, class_num)
        P = P.permute([1, 0, 2, 3])
        # (seq_length, batch_size, class_num, class_num)
        return P

    def p_down(self, P, length):
        """ 计算概率和
        @param:
        :P: (seq_length, batch_size, class_num, class_num)
        :length: (batch_size)
        """
        P = torch.exp(P)
        p_down = P[0, :, self._label_start_, :].unsqueeze(dim=1) # (batch_size, 1, class_num)
        p_sums = [p_down]
        for P_t in P[1:]: # (batch_size, class_num, class_num)
            p_down = torch.matmul(p_down, P_t) # (batch_size, 1, class_num)
            p_sums.append(p_down)
        p_final = []
        for i, l in enumerate(length):
            p_final.append(p_sums[l - 1][i, :,:]) # (1, 1, class_num)
        p_final = torch.cat(p_final, dim=0) # (batch_size, class_num)
        p_ans = torch.sum(p_final, dim=1) # (batch_size)
        p_down = torch.log(p_ans)
        return p_down
    
    def p_up(self, P, length, label):
        """ 计算序列概率
        @param:
        :P: (seq_length, batch_size, class_num, class_num)
        :length: (batch_size)
        :label: (seq_length, bacth_size)
        """
        p_up_t = [class_[self._label_start_, label[0, index]] for index, class_ in enumerate(P[0])]
        p_up_t = torch.stack(p_up_t)
        p_ups = [p_up_t]
        for t, P_t in enumerate(P[1:]): # (batch_size, class_num, class_num)
            p_up_t = [class_[label[t, index], label[t + 1, index]] for index, class_ in enumerate(P_t)]
            p_up_t = torch.stack(p_up_t)
            p_ups.append(p_up_t)
        p_ups = torch.stack(p_ups) # (seq_length, batch_size)
        p_ups = p_ups.permute([1, 0])
        p_up = []
        for index, p in enumerate(p_ups):
            p_up.append(torch.sum(p[:length[index]]))
        p_up = torch.stack(p_up)
        return p_up

    def predict(self, P):
        """ P各个时间步的转移概率
        @param:
        :P: (seq_length, batch_size, class_num, class_num)
        """
        P = torch.exp(P)
        P = P.permute([1, 0, 2, 3])
        batch_label_list = []
        for p_batch in P:
            p_t = p_batch[0, self._label_start_]
            value_t, label_t = torch.topk(p_t, self.k)
            label_list = [[label_t[i]] for i in range(self.k)]
            value_list = value_t
            for p_t in p_batch[1:]: # (class_num, class_num)
                label_list_ = []
                value_list_ = []
                for index, label_value in enumerate(zip(label_t, value_t)):
                    label_t_i, value_t_i = label_value
                    value_t_, label_t_ = torch.topk(p_t[label_t_i], self.k)
                    value_t_ = value_t_*value_t_i
                    label_list_.append(label_t_)
                    value_list_.append(value_t_)
                label_list_ = torch.cat(label_list_) # (k*k)
                value_list_ = torch.cat(value_list_) # (k*k)
                value_list_, topk_label = torch.topk(value_list_, self.k)
                label_list_new = []
                for topk_label_i in topk_label: # topk的label
                    label_list_new.append(copy.deepcopy(label_list[topk_label_i//self.k]) + [label_list_[topk_label_i]])
                label_list = label_list_new
            value, indice = torch.max(value_list, dim=0)
            batch_label_list.append(torch.stack(label_list[indice]))
        batch_label_list = torch.stack(batch_label_list)
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
    crf = CRF(hidden_size=hidden_size, class_num=class_num, device=torch.device('cpu'), label_start=0, k=5)
    print(crf.predict(P))