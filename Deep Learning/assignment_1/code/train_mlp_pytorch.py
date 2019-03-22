"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  total = predictions.shape[0]
  _, predicted = torch.max(predictions, 1)
  correct = 0
  correct += ( predicted == targets).sum().item()
  #print("correct number "+ str(correct))
  accuracy = correct/total
  #print("accuracy "+ str(accuracy))
  #raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

 
  # Using parameters from the inputs
  Learning_rate = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq  
  
  # Get data from directory
  cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  
  # Set up batch size
  x,y = cifar10['train'].next_batch(batch_size)
  x=x.reshape(batch_size, 3*32*32)
  x=torch.from_numpy(x)
  y=torch.from_numpy(y)
  _,y=torch.max(y,1)
  #print(y)

  # Get test data
  x_t,y_t = cifar10['test'].images, cifar10['test'].labels
  x_t=x_t.reshape(x_t.shape[0],3*32*32)
  x_t=torch.from_numpy(x_t[0:1000])
  y_t=torch.from_numpy(y_t[0:1000])
  _,y_t=torch.max(y_t,1)
  # Construct the mlp
  n_hidden_node = dnn_hidden_units
  n_inputs_size = 3*32*32 
  n_classes = 10
  mlp = MLP(n_inputs_size, n_hidden_node, n_classes)
  Out = mlp.forward(x)

  # Update gradients of linear module params
  optimizer = optim.SGD(mlp.parameters(), lr=Learning_rate)
  # Calculate cross entropy value
  CM = nn.CrossEntropyLoss()
  L = CM(Out, y)
  # Evaluate the results
  accuracy_list = []
  loss = []

  #print("prediction" + str(P))
  #print("targets" + str(y_t[0:5]))

  a = accuracy(Out, y)

  #print("accuracy rate" + str(a))
  accuracy_list.append(a)


  # Start training
  step=0

  while (step <= max_steps):
      print("step number: "+ str(step))
      x,y = cifar10['train'].next_batch(batch_size)
      x=x.reshape(batch_size, n_inputs_size)
      x=torch.from_numpy(x)
      y=torch.from_numpy(y)
      _,y=torch.max(y,1)
      # Zero the gradient buffers
      optimizer.zero_grad() 
      # Forward propagation
      out = mlp.forward(x)
      # calculate the loss
      L = CM(out, y)
      loss.append(L)
    
      # in your training loop:

      L.backward()
      optimizer.step()    

      # Evaluation
      if (step%eval_freq) ==0: 
          p = mlp.forward(x_t)
          a = accuracy(p, y_t)
          print("The accuracy rate of step " + str(step) + " is " + str(a))
          accuracy_list.append(a)
      step+=1
  plt.figure()
  plt.plot(list(range(1, len(loss)+1)), loss)
  plt.ylabel("Training loss at each step")
  plt.xlabel("Training steps")
  plt.title("Cross Entropy loss at each step")
  plt.show()

  plt.figure()
  plt.plot(list(range(1, len(accuracy_list)+1)), accuracy_list)
  plt.ylabel("Accuracy rate")
  plt.xlabel("Training steps")
  plt.title("Accuracy rate at each 100 steps")
  plt.show()

 # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
