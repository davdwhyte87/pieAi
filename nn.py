import numpy as np

x = np.random.rand(784,256)
output = np.random.rand(784, 10)
w1 = np.random.rand(256, 126)
b1 = np.random.rand(126)
h1 = np.dot(x,w1)+b1
w2 = np.random.rand(126, 10)
b2 = np.random.rand(10)
y = np.dot(h1, w2)+b2
def sigmoid(z):
  """The sigmoid function."""
  return 1.0/(1.0+np.exp(-z))
print(y.shape)
celoss = ((y-output) * output * (1 - output))
print(np.mean(np.square(celoss)))
print(sigmoid(y).shape)
# for i in range(y):
#   if maxy[i] == output[i]:
#     accuracy = 100/
class Network():
  def __init__(self):
    self.x = np.random.rand(784,256)
    self.y = np.random.rand(784, 10)
    self.w1 = np.random.rand(256, 126)
    self.b1 = np.random.rand(126)
    self.w2 = np.random.rand(126, 10)
    self.b2 = np.random.rand(10)
  def cross_entropy_loss(output):
    return (self.y-output) * output * (1 - output)
  def forward(self):
    # the forward function gets the data and passes it through all the layers
    h1 = np.dot(self.x,self.w1)+self.b1
    y = np.dot(h1, self.w2)+self.b2
    return y
  def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
  # def train():

  
net = Network()
h = net.forward()
print(h.shape)

