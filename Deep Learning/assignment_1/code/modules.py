"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print('in_features: '+str(in_features))

    self.params = {'weight': None, 'bias': None}
    self.params['weight'] = np.random.normal(0,0.0001, (out_features,in_features))
    self.params['bias'] = np.zeros((out_features))
    self.grads = {'weight': None, 'bias': None}
    self.grads['weight'] = np.zeros((out_features,in_features))
    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    ####################### 
   # print(self.params['weight'].shape)

    out = np.dot(x, self.params['weight'].T) + self.params['bias'] 
    self.x = x.copy()
    # print(out.shape)
 #   raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################	
    #Get the dimension of each matrix
   # print("shape of dout: " + str(dout.shape))
    

    b = dout.shape[0]
    c = dout.shape[1]
    n = self.x.shape[1]

    #reshape x matrix and dout
    #x_in=self.x.reshape(b,n,1)
    #dout_1=dout.reshape(b,1,c)
    #self.grads['weight'] = (np.sum(np.matmul(x_in,dout_1), axis = 0)/b).T
    self.grads['weight'] = np.dot(dout.T, self.x)
    #print("Gradient shape: "+ str(self.grads['weight']))
    self.grads['bias'] = np.sum(dout, axis=0)
    #print("Bias shape: "+ str(self.grads['bias']))

    dx = np.dot(dout, self.params['weight']) 
    #print("dx "+ str(dx))
  #  raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print(x.shape)
    
    self.x=x
    x_tilde = x.copy()
    dx = x.copy()
    # Check dimension
    if len(x.shape) == 1:
        for i in range(x.shape[0]):
            if x[i]>=0:
                x_tilde[i] = x[i]
                dx[i] = 1
            else: 
                x_tilde[i] = 0
                dx[i] = 0 
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j]>=0:
                    x_tilde[i][j]=x[i][j]
                    dx[i][j]=1
                else:
                    x_tilde[i][j]=0
                    dx[i][j]=0
    self.dx = dx
    out = x_tilde
    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
  #  print("shape of dout: " + str(dout.shape))
    b=dout.shape[0]
    out=dout*self.dx
    dx = out
  #  print("shape of dx: " + str(dx.shape))
 #   print("dx "+ str(dx))
  #  raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    x_exp = x.copy()
    #print("shape of x: " + str(x.shape))
    for i in range(x.shape[0]):
        row_max=x[i].max()
        for j in range(x.shape[1]):
            x_exp[i][j]=np.exp(x[i][j]-row_max)
    exp_sum=np.sum(x_exp,axis=1).reshape(x.shape[0],1)
    #print("x_exp "+ str(x_exp))
    out = x_exp/exp_sum
   
    self.exp_sum=exp_sum
    self.out = out
  #  print("out: "+ str(out))
   # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b=dout.shape[0]

    #print("b"+str(b))
    #print("dout"+str(dout))
    out = self.out
    #print("shape of out "+ str(self.out.shape))
    module_gradient=np.zeros((dout.shape[0], dout.shape[1], dout.shape[1]))

    for k in range(dout.shape[0]):
        for i in range(dout.shape[1]):
            for j in range(dout.shape[1]):
                if i==j:
                    module_gradient[k][i][j] = out[k][i]*(1-out[k][j])
                else:
                    module_gradient[k][i][j] = -out[k][j]*out[k][i]
    dout = dout.reshape(dout.shape[0],1, dout.shape[1])

    dx = np.matmul(dout, module_gradient)
    dx = dx.reshape(dout.shape[0], dout.shape[2])
   # print("dx "+ str(dx))
    #print("shape of dx "+ str(dx.shape))
   # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b = x.shape[0]
    info = x*y
    info = info[info>0]
    out = np.sum(-np.log(info))/b
    #out = info
    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b = x.shape[0]
  #  print(x)
    index = np.argmax(x, axis=1)
   #print()
 #   dx = np.zeros((x.shape[0], x.shape[1])) 
    dx = -((np.divide(y,x)) + 0)/b  
  #  dx = dx/x.shape[0]
#    print("shape of dx " + str(dx.shape))
   # for i in range(x.shape[0]):
    #    for j in range(x.shape[1]):
     #       if index[i] == j:
      #          dx[i][j] = -1/x[i][j]
       #     else:
        #        dx[i][j] = 0
    
   # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
