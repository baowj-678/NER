""" 
@Paper: End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/22
"""

import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.nn.utils.rnn import pack_padded_sequence 


class CNN(nn.Module):
    def __init__(self, 
                 device, 
                 char_vocab, 
                 char_embedding_dim, 
                 p=0.1, 
                 kernel_n=5, 
                 padding=2,
                 max_sent_length=20, 
                 max_word_length=15):
        super(CNN, self).__init__()
        """ CNN初始化
        """
        # character level vocab
        self.char_vocab = char_vocab
        # device
        self.device = device
        # padding
        self.padding = char_vocab.pad_i
        # max sentence length
        self.max_sent_length = max_sent_length
        # max word length
        self.max_word_length = max_word_length
        # character embedding dim
        self.char_embedding_dim = char_embedding_dim
        # Embedding
        Sampler = tdist.Normal(torch.tensor(0.0), torch.tensor((3/char_embedding_dim)**0.5))
        self.char_embedding = nn.Parameter(Sampler.sample(sample_shape=(len(char_vocab), char_embedding_dim)), requires_grad=True).to(device)
        # dropout
        self.dropout = nn.Dropout(p=p)
        # convolution
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_n, 1), stride=1, padding=(padding, 0))
        # max pooling
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, data):
        """ 卷积\n
        @param:\n
        :data (batch_size, sent_length)[str]\n
        @return:\n
        :data_embedded (batch_size, sent_length, hidden_size), the embedded word of every sentence.\n
        :length (batch_size), the lengths for every sentence of batch.\n
        """
        print(self.conv.weight)
        data, length = self.embedding(data)
        # data [tensor](batch_size, max_sent_length, max_word_length, char_embedding_size)
        # length [tensor](batch_size)
        data = self.dropout(data)
        batch_size, max_sent_length = data.shape[:2]
        data = data.unsqueeze(dim=2).flatten(start_dim=0, end_dim=1)
        # (batch_size * max_sent_length, 1, max_word_length, char_embedding_size)
        data_conv = self.conv(data).squeeze(dim=1)
        # (batch_size * max_sent_length, max_word_length + padding*2 - kernel_n + 1, char_embedding_size)
        data_permute = data_conv.permute([0, 2, 1])
        # (batch_size * max_sent_length, char_embedding_size, max_word_length + padding*2 - kernel_n + 1)
        data_pool = self.max_pooling(data_permute).squeeze(dim=2)
        # (batch_size * max_sent_length, char_embedding_size)
        data_embedded = data_pool.reshape(batch_size, max_sent_length, -1)
        # (batch_size, max_sent_length, char_embedding_size)
        return (data_embedded, length)
    
    def embedding(self, data):
        """ character-level embedding\n
        @parameter:\n
        :data [list](batch_size, sent_length):[str], source without embedding\n
        @return:\n
        :embedding_data (batch_size, sent_length, max_sent_length, char_embedding_dim)\n
        :length [list](batch_size)[int], sentence length for every batch\n
        """
        batch_size = len(data)
        embedding_data = torch.zeros(size=(batch_size, self.max_sent_length, self.max_word_length, self.char_embedding_dim))
        length = []
        for i, sent in enumerate(data):
            length_tmp = min(len(sent), self.max_sent_length)
            length.append(length_tmp)
            for j, word in enumerate(sent[:length_tmp]):
                if len(word) > self.max_word_length:
                    for k, c in enumerate(word[:self.max_word_length]):
                        embedding_data[i, j, k] = self.char_embedding[self.char_vocab[c]]
                else:
                    for k, c in enumerate(word):
                        embedding_data[i, j, k] = self.char_embedding[self.char_vocab[c]]
                    for k in range(self.max_word_length - len(word)):
                        embedding_data[i, j, k + len(word)] = self.char_embedding[self.padding]
        embedding_data = embedding_data.to(self.device)
        length = torch.tensor(length, device=self.device)
        return (embedding_data, length)
    
    def save_gradient(self, Model, grad_i, grad_o):
        """ 保存梯度
        """
        self.gradients = {Model, (grad_i, grad_o)}
        


if __name__ == "__main__":
    cnn = CNN(torch.device('cpu'), char_embedding_dim=5)
    embedded_char = [['hello', 'world'], ['i', 'am', 'student']]
    print(cnn(embedded_char))

    # m = nn.AdaptiveMaxPool1d(1)
    # input = torch.randn(4, 64, 8)
    # output = m(input)
    # print(output.shape)