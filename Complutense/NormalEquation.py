import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv

data = read_csv('p1/ex1data2.csv', header=None).values.astype(float)
x = data[:,:2]
y = data[:,2]
size = y.size
y.shape = (size, 1)

X = np.ones((size,3))
X[:,1:3] = x

def normalEquation(X, y):
    return np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))

normalEquationTheta = normalEquation(X, y)
normalEquationPrediction = np.array([1.0, 1650.0, 3]).dot(normalEquationTheta)
print("Normal equation prediction: %s" % normalEquationPrediction)
