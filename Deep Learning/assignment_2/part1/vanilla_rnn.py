################################################################################
# MIT License
#
# Copyright (c) 2018
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

import dataset
import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
    #    self.params = {'w_in': None,'w_state': None,'w_out': None,'b_in': None,'b_out': None}
        #self.h = torch.zeros((num_hidden, batch_size))
        self.l = seq_length
        self.D = input_dim
        self.f = num_hidden
        self.c = num_classes
        self.b = batch_size
      #  p = dataset.PalindromeDataset()
      #  self.x = p.generate_palindrome()
        self.step = 0
        self.w_in = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.w_state =nn.Parameter(torch.randn((num_hidden, num_hidden)))
        self.w_out = nn.Parameter(torch.randn((num_hidden,num_classes)))
        self.b_in = nn.Parameter(torch.randn(num_hidden,1))
        self.b_out = nn.Parameter(torch.randn(1,num_classes))

    def forward(self, x):
        # Implementation here ...
        #print(type(x))
        h = torch.zeros((self.f, self.b))
        for t in range(self.l):
        #    print(t)
        #    print(torch.mm(self.w_in,x[:,t].reshape(1,-1)))
            linear = torch.mm(self.w_in,(x[:,t].reshape(1,-1))) + torch.mm(self.w_state,h) + self.b_in
            h = torch.tanh(linear)
            p = torch.mm(torch.t(h),self.w_out) + self.b_out
        #    y = nn.softmax(p)
        #    predict = y.max(0)[1]

        return p
