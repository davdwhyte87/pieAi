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
# print(y.shape)
celoss = ((y-output) * output * (1 - output))
# print(np.mean(np.square(celoss)))
# print(sigmoid(y).shape)


def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))

# for i in range(y):
#   if maxy[i] == output[i]:
#     accuracy = 100/
class Network():
  def __init__(self):
    self.x = np.random.rand(784,)
    self.y = np.random.rand(784, 10)
    self.h = 0
    self.z = 0
    self.w1 = np.random.rand(784, 126)
    self.b1 = np.random.rand(126)
    self.w2 = np.random.rand(126, 10)
    self.b2 = np.random.rand(10)

    self.l1 = 0
    self.new_w_changes = []
    self.new_l1_changes = []
    self.new_b_changes = []
  def cross_entropy_loss(output):
    return (self.y-output) * output * (1 - output)

  def simple_cost(self, output):
    return (self.y-output)**2

  def forward(self):
    # the forward function gets the data and passes it through all the layers
    self.h = np.dot(self.x,self.w1)+self.b1
    # self.z = np.dot(self.h, self.w2)+self.b2
    # self.a = sigmoid(self.z)
    # self.back_prop()
    print(self.x.shape)
    return self.h

  def back_prop(self):
    dC_da = np.dot(2, (self.simple_cost(self.a)))
    da_dz = sigmoid_prime(self.z)
    dz_dw = self.z

    dC_dw = dC_da*da_dz*dz_dw
    # print(dC_dw[0])
    # print(self.w1.shape)

  def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
  # def train():

net = Network()
h = net.forward()
# print(h.shape)

