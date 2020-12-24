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
                 char_vocab,
                 entity_vocab,
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
                 max_word_length=20,
                 dropout=0.1):
        super(Model, self).__init__()
        """ 初始化
        :device: 训练设备
        :char_embedding: CNN中char embedding的size
        :kernal_n: CNN卷积核大小
        :padding: CNN卷积padding大小
        :hidden_size: BiLSTM隐藏层大小
        """
        # Device
        self.device = device
        # Max sentence length
        self.max_sent_length = max_sent_length
        # Vocab
        self.vocab = vocab
        # CNN for Character-level Representation
        self.cnn = CNN(device=device,
                       char_vocab=char_vocab,
                       char_embedding_dim=char_embedding_dim,
                       p=0.1,
                       kernel_n=kernel_n,
                       padding=padding,
                       max_sent_length=max_sent_length,
                       max_word_length=max_word_length).to(device)
        # word Embedding
        self.word_embedding = load_glove(glove_file, word2vec_file, vocab, word_embedding_dim, device=device)
        # BiLSTM
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=char_embedding_dim + word_embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            # dropout=dropout,
                            bidirectional=True).to(device)
        # CRF
        self.crf = CRF(hidden_size=hidden_size*2, # LSTM是双向的
                       class_num=class_num,
                       device=device,
                       entity_vocab=entity_vocab,
                       k=3).to(device)

    
    def forward(self, input_data, target_entity=None):
        """
        @param:\n
        :input_data list[str]: (batch_size, sent_length),按照句长降序排列\n
        :target list[int]: \n
        """
        """ CNN """
        # if target_entity is not None:
        #     target_entity = torch.tensor(target_entity, device=self.device)
        #     target_entity = target_entity.permute([1, 0])
        #     # (seq_length, batch_size)
        input_data_char_embeded, input_data_length = self.cnn(input_data) 
        # (batch_size, sent_length, char_embeded_size)
        # (batch_size)
        input_data_index, input_data_length_ = self.vocab.wordlists2index(input_data, self.max_sent_length)
        input_data_index = torch.tensor(input_data_index, device=self.device)
        input_data_length_ = torch.tensor(input_data_length_, device=self.device)
        # [tensor](batch_size, sent_length)
        # [tensor](batch_size)
        assert torch.equal(input_data_length, input_data_length_)
        input_data_word_embeded = self.word_embedding(input_data_index)
        # (batch_size, sent_length, word_embeded_size)
        input_data_embeded = torch.cat((input_data_char_embeded, input_data_word_embeded), dim=2)
        # (batch_size, sent_length, hidden_size (word_embeded_size + char_embeded_size))
        # print('CNN finish')
        """ LSTM """
        input_data_packed = pack_padded_sequence(input_data_embeded, input_data_length, batch_first=True)
        data_lstm, hidden_state = self.lstm(input_data_packed)
        # print('BiLSTM finish')
        """ CRF """
        loss = self.crf(input=data_lstm, entity=target_entity)
        # print('CRF finish')
        return loss
        



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
        
