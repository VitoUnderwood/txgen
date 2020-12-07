# -*- coding:utf-8 -*-
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # input_size 词汇表的大小，hidden size 词向量的大小,与输入的尺寸没有必然联系，输入的id应该在input size的范围内
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True,
                          bidirectional=False)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, hidden):
        emb = self.encoder(input_ids)
        output, hidden = self.gru(emb, hidden)
        output = self.decoder(output[:, -1, :])
        return output

class Test:
    def __init__(self):
        print('test')
