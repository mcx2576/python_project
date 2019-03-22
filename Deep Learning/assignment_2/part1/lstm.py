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

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        # Initialize the the parameter object class
        #self.params = {'w_gx': None, 'w_gh': None, 'w_ix': None, 'w_ih': None, 'w_fx': None, 'w_fh': None,  'w_ph': None,
        #               'w_ox': None, 'w_oh': None,'b_g': None,'b_i': None,'b_f': None, 'b_o': None, 'b_p': None}
        # Initialize the other objects
    #    self.h = torch.zeros((num_hidden, batch_size))
    #    self.C = torch.zeros((num_hidden, batch_size)) #Or the batch size, not sure
        self.l = seq_length
        self.D = input_dim
        self.f = num_hidden
        self.c = num_classes
        self.b = batch_size
    #    self.step = 0

        # Define each object within the parameter object
        self.w_gx = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.w_gh = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.w_ix = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.w_ih = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.w_fx = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.w_fh = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.w_ox = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.w_oh = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.w_ph = nn.Parameter(torch.randn(num_hidden,num_classes))

        self.b_g = nn.Parameter(torch.randn(num_hidden))
        self.b_i = nn.Parameter(torch.randn(num_hidden))
        self.b_f = nn.Parameter(torch.randn(num_hidden))
        self.b_o = nn.Parameter(torch.randn(num_hidden))
        self.b_p = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        # Updatet the current step
        h = torch.zeros((self.f, self.b))
        C = torch.zeros((self.f, self.b))
        #Perform the calculation
        for t in range(self.l):
            g = torch.tanh(torch.mm(self.w_gx,x[:,t].reshape(1,-1)) + torch.mm(self.w_gh,h) + self.b_g )
            #print(torch.mm(self.w_ix,x[:,t].reshape(1,-1)) + torch.mm(self.w_ih,h) + self.b_i )
            i = torch.sigmoid(torch.mm(self.w_ix,x[:,t].reshape(1,-1)) + torch.mm(self.w_ih,h) + self.b_i )
            f = torch.sigmoid(torch.mm(self.w_fx,x[:,t].reshape(1,-1)) + torch.mm(self.w_fh,h) + self.b_f )
            o = torch.sigmoid(torch.mm(self.w_ox,x[:,t].reshape(1,-1)) + torch.mm(self.w_oh,h) + self.b_o )
            C = torch.mul(g,i)+torch.mul(C,f)
            h = torch.mul(torch.tanh(C), o)

            p = torch.mm(torch.t(h),self.w_ph) + self.b_p
        #    y = nn.softmax(p)
        #    predict = y.max(0)[1]

        return p
