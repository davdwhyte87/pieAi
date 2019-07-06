import numpy as np


class Net:
  def __init__(self):
    self.W1 = np.random.rand(784, 256)
    self.X = np.random.rand(784)
    self.Y = np.random.rand(10)
    self.B1 = np.random.rand(256)

    self.H1 = 0
    self.W2 = np.random.rand(256, 10)
    self.H2 = 0
    self.B2 = np.random.rand(10)
    self.A = 0

  def forward(self):
    self.H1 = np.dot(self.X, self.W1) + self.B1
    self.H2 = np.dot(self.H1, self.W2) + self.B2
    self.A = self.sigmoid(self.H2)
    # self.backprop()
    return self.A


  def simple_cost(self, output):
    return (self.Y-output)**2

  def sigmoid(self,z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

  def sigmoid_prime(self, z):
    return self.sigmoid(z)*(1-self.sigmoid(z))

  def backprop(self):
    #back prop for W2 values 
    dC_dA = np.dot(2,(self.simple_cost(self.A)))
    dA_dH2 = self.sigmoid_prime(self.H2)
    dH2_dW2 = self.H1
    dC_dW2 =  dC_dA* dA_dH2* np.transpose(np.array([dH2_dW2,]))

    #back prop for B2 values 
    dH2_dB2 = self.W2 * np.transpose(np.array([self.H1,]))
    ess =  dC_dA* dA_dH2
    dC_dB2  =  ess* np.transpose(np.array([dH2_dB2,]))
    dC_dB2 = dC_dB2[0,:,:]


    #back prop for W1 values 
    # butom = 1/(np.square(self.H1)) 
    # top = (-1* (self.B2-self.H2))
    # dW2_dH1 = top * np.transpose(np.array([butom,]))

  
    from numpy import newaxis
    newH1 = self.H1[:,newaxis]
    dW2_dH1 = np.dot(self.H1, self.W2)
    print(dW2_dH1.shape)
    dH1_dW1 = self.X

    blue = np.dot(dC_dW2,dW2_dH1)
    dC_dW1 = blue* np.transpose(np.array([dH1_dW1,]))
    print(dC_dW1.shape)

    # dC_dW2XdW2_dH1 = dC_dW2 * dW2_dH1 
    # dC_dW1 = self.X * np.transpose(np.array([dC_dW2XdW2_dH1,]))
    # dC_dW1 = dC_dW1.T
    # dC_dW1 = dC_dW1[:,:,0]
    # print((dC_dW1).shape)

    #back prop for B1 values
    dH1_dB1 = self.B1
    dB2_dH1 = self.B2 * np.transpose(np.array([self.H1,]))
    blue2 = np.dot(dC_dB2.T,dB2_dH1)
    print(blue2.shape)
    dC_dB1 = dH1_dB1 * np.transpose(np.array([blue2,]))
    print(dC_dB1.shape)


net = Net()
predictions = net.forward()
# print(predictions.shape)