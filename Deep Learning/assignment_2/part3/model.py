# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        # rnn = nn.LSTM(10, 20, 2)
        # input = torch.randn(5, 3, 10)
        # h0 = torch.randn(2, 3, 20)
        # c0 = torch.randn(2, 3, 20)
        # output, (hn, cn) = rnn(input, (h0, c0))
        # input_size, hidden_size , num_layers, bias
        self.rnn = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.f = lstm_num_hidden
        self.b = batch_size
        self.l = seq_length
        self.n_l = lstm_num_layers

    def forward(self, x):
        # Implementation here...
        h = torch.zeros(self.n_l, self.b, self.f)
        C = torch.zeros(self.n_l, self.b, self.f )
        output, (hn, cn) = self.rnn(x, (h, C))
        out = self.linear(output)

        return out
