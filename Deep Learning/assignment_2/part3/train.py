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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def train(config):

    # Initialize the device which to run the model on
    #device = torch.device(config.device)


    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel( config.batch_size, config.seq_length, dataset._vocab_size, config.lstm_num_hidden, config.lstm_num_layers)  # fixme

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate )
   # optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate ) 
    #  # fixme

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
       # if batch_inputs[0].shape[0] != config.batch_size:
        #    continue
        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        optimizer.zero_grad()
        def one_hot(x, num_classes):
            for t in range(len(x)):
                y = torch.eye(num_classes)
                x[t]=y[x[t]]
            return  torch.stack(x)

        batch_inputs = one_hot(batch_inputs,  dataset._vocab_size)
        predicts = model.forward(batch_inputs)
        
        
        # Calculate average loss
        loss=[]
    #    print(predicts.shape)
        batch_targets = torch.stack(batch_targets)
        for t in range(config.seq_length):
            l  = criterion(predicts[t], batch_targets[t])
            loss.append(l)
        loss = sum(loss)/config.seq_length # fixme
        #print(type(float(loss.detach())))
        loss.backward()
        optimizer.step()
    #    loss = float(loss.detach())

        accuracy = [] # fixme
        _, predicted = torch.max(predicts, 2)
       # print(batch_targets[:,1].shape)
      #  print(predicted[:,1].shape)
        for b in range(config.batch_size):
            correct = 0
            correct += ( predicted[:,b] == batch_targets[:,b]).sum().item()
            avg_accuracy = correct /config.seq_length
            accuracy.append(avg_accuracy)

        accuracy =sum(accuracy)/config.batch_size
    #    print(type(accuracy))
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
        #print(type(config.train_steps))
        #print(type(step))
        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            sentence=[]
            a =  np.random.randint(dataset._vocab_size, size=1)
            first = torch.eye(dataset._vocab_size)[a].unsqueeze_(0)
            sentence.append(dataset._ix_to_char[int(a)])
            h=torch.zeros(model.n_l,1,model.f)
            c=torch.zeros(model.n_l,1,model.f)
         #   for t in range(config.seq_length):
                #print(t)
            #   output, (h, c) = model.rnn(first, (h,c))
             #   out = model.linear(output)
                # Deal with the temperature paramenter
             #   if config.temperature ==1: #or step ==0:
              #      _, b= torch.max(out, 2)
              #  else: 
              #      predicts = torch.exp(out/config.temperature)
              #      x = torch.distributions.Categorical(predicts)
                    
               #     b = x.sample()
               #     print(b)
                #_, b=torch.max(out, 2)
             #   sentence.append(dataset._ix_to_char[int(b)])
              #  first = torch.eye(dataset._vocab_size)[b]
            print('GS: '+str(''.join(str(i) for i in sentence)))
            pass
        
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            torch.save(model.state_dict(), "model.pth")
            break


    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--temperature', type=float, default=1, help='If not greedy sampled')

    config = parser.parse_args()

    # Train the model
    train(config)
