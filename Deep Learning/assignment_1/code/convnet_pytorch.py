"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, (3, 3), stride = 1, padding = 1)
    self.batch1 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d((3, 3), stride = 2, padding = 1)
    self.conv2 = nn.Conv2d(64, 128, (3, 3), stride = 1, padding = 1)
    self.batch2 = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d((3, 3), stride = 2, padding = 1)
    self.conv3a = nn.Conv2d(128, 256, (3, 3), stride = 1, padding = 1)
    self.batch3a = nn.BatchNorm2d(256)
    self.conv3b = nn.Conv2d(256, 256, (3, 3), stride = 1, padding = 1)
    self.batch3b = nn.BatchNorm2d(256)
    self.pool3 = nn.MaxPool2d((3, 3), stride = 2, padding = 1)
    self.conv4a = nn.Conv2d(256, 512, (3, 3), stride = 1, padding = 1)
    self.batch4a = nn.BatchNorm2d(512)
    self.conv4b = nn.Conv2d(512, 512, (3, 3), stride = 1, padding = 1)
    self.batch4b = nn.BatchNorm2d(512)
    self.pool4 = nn.MaxPool2d((3, 3), stride = 2, padding = 1)
    self.conv5a = nn.Conv2d(512, 512, (3, 3), stride = 1, padding = 1)
    self.batch5a = nn.BatchNorm2d(512)
    self.conv5b = nn.Conv2d(512, 512, (3, 3), stride = 1, padding = 1)
    self.batch5b = nn.BatchNorm2d(512)
    self.pool5 = nn.MaxPool2d((3, 3), stride = 2, padding = 1)
    self.avgpool = nn.AvgPool2d((1,1), stride = 1, padding = 0)
    self.fc1 = nn.Linear(512, 10)

    


 #   raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    x = self.pool1(F.relu(self.batch1(self.conv1(x))))
    x = self.pool2(F.relu(self.batch2(self.conv2(x))))
    x = F.relu(self.batch3a(self.conv3a(x)))
    x = self.pool3(F.relu(self.batch3b(self.conv3b(x))))
    x = F.relu(self.batch4a(self.conv4a(x)))
    x = self.pool4(F.relu(self.batch4b(self.conv4b(x))))
    x = F.relu(self.batch5a(self.conv5a(x)))
    x = self.pool5(F.relu(self.batch5b(self.conv5b(x))))
    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    out = self.fc1(x)

#    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
