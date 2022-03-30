import numpy as np;

learning_rate=1.5

X = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
Y = np.array([[1, 0, 1]])

m = Y.shape[1]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost
    

for iter in range(25000):
  z=np.dot(w.t,X)+b
  A=sigmoid(z)
  dz=A-Y
  dW = (1 / m) * np.dot(dz, X.T)
  db = (1 / m) * np.sum(dz)
  w= w - learning_rate * dW
  b= b - learning_rate * db
  print(w)