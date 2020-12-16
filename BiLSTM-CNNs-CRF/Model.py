from typing import Optional
import torch
import torch.nn as nn
from CNN import CNN
from CRF import CRF
from LoadGloVe import load_glove
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
from dataset import Vocab

class Model(nn.Module):
    def __init__(self, 
                 vocab,
                 device=None, 
                 char_embedding_dim=128,
                 word_embedding_dim=50,
                 hidden_size=128,
                 kernel_n=5, 
                 padding=2,
                 class_num=4,
                 glove_file: Optional[str]=None,
                 word2vec_file: str='',
                 max_sent_length=20,
                 dropout=0.1):
        super(Model, self).__init__()
        """ 初始化
        :device: 训练设备
        :char_embedding: CNN中char embedding的size
        :kernal_n: CNN卷积核大小
        :padding: CNN卷积padding大小
        :hidden_size: BiLSTM隐藏层大小
        """
        # Max sentence length
        self.max_sent_length = max_sent_length
        # Vocab
        self.vocab = vocab
        # CNN for Character-level Representation
        self.cnn = CNN(device=device,
                       char_embedding_dim=char_embedding_dim,
                       p=0.1,
                       kernel_n=kernel_n,
                       padding=padding,
                       char_num=30,
                       char_padding_id=0,
                       max_sent_length=20,
                       max_word_length=15)
        # word Embedding
        self.word_embedding = load_glove(glove_file, word2vec_file, vocab, word_embedding_dim)
        # BiLSTM
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=char_embedding_dim + word_embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            # dropout=dropout,
                            bidirectional=True)
        
        # CRF
        self.crf = CRF(hidden_size=self.hidden_size,
                       class_num=class_num,
                       device=device,
                       label_start=0,
                       k=3)

    
    def forward(self, input_data, target=None):
        """
        @param:\n
        :input_data list[str]: (batch_size, sent_length),按照句长降序排列\n
        :target list[int]: \n
        """
        """ CNN """
        input_data_char_embeded, input_data_length = self.cnn(input_data) 
        # (batch_size, sent_length, char_embeded_size)
        # (batch_size)
        input_data_index, input_data_length_ = self.vocab.wordlists2index(input_data, self.max_sent_length)
        # [list](batch_size, sent_length)
        # (batch_size)
        input_data_index = torch.tensor(input_data_index) # tensor
        assert input_data_length == input_data_length_
        input_data_word_embeded = self.word_embedding(input_data_index)
        # (batch_size, sent_length, word_embeded_size)
        input_data_embeded = torch.cat((input_data_char_embeded, input_data_word_embeded), dim=2)
        # (batch_size, sent_length, hidden_size (word_embeded_size + char_embeded_size))

        """ LSTM """
        input_data_packed = pack_padded_sequence(input_data_embeded, input_data_length, batch_first=True)
        data_lstm, hidden_state = self.lstm(input_data_packed)
        # print(data_lstm.shape)
        """ CRF """
        return data_lstm, hidden_state
        



if __name__ == '__main__':
    path = 'D:/NLP/NER/dataset/CONLL2003/vocab.txt'
    vocab = Vocab(path)
    device = torch.device('cuda')
    model = Model(vocab = vocab,
                  device = device, 
                  char_embedding_dim=128,
                  word_embedding_dim=50,
                  hidden_size=128,
                  kernel_n=5,
                  padding=2,
                  class_num=4,
                  glove_file=None,
                  word2vec_file='D:/NLP/Word2Vec/GloVe/glove6B50d.txt',
                  max_sent_length=20,
                  dropout=0.1)
    data = [['I', 'am', 'a', 'student'], ['hello', 'world']]
    model(data)
        
