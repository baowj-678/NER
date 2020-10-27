import torch
import torch.nn as nn
from CNN import CNN

class Model(nn.Module):
    def __init__(self, 
                 device=None, 
                 char_embedding=128, 
                 kernal_n=5, 
                 padding=2,
                 hidden_size=128,
                 dropout =0.1):
        super(Model, self).__init__()
        """ 初始化
        :device: 训练设备
        :char_embedding: CNN中char embedding的size
        :kernal_n: CNN卷积核大小
        :padding: CNN卷积padding大小
        :hidden_size: BiLSTM隐藏层大小
        """
        # CNN for Character-level Representation
        # Embedding
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=char_embedding, padding_idx=0)
        # CNN
        self.cnn = CNN(device=device, embedding_size=char_embedding, n=kernal_n, padding=padding)

        # BiLSTM
        self.lstm = nn.LSTM(input_size=char_embedding,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
    
    def forward(self, input_data, target=None):
        """
        :input_data list[str]: (batch_size, sent_length)
        :target list[int]: 
        """
