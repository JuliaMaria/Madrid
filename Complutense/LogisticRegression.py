import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv
import scipy.optimize as opt

data = read_csv('p2/ex2data1.csv', header=None).values.astype(float)
x = data[:,:2]
y = data[:,2]
size = len(y)
positive = plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], marker='+', c='k')
negative = plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], marker='o', c='b')
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend((positive, negative),('Admitted', 'Not admitted'))
plt.show()

X = np.ones((size,3))
X[:,1:] = x

theta = np.zeros((3, 1))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cost_function(theta, x, y, m):
    J = (-np.log(sigmoid(x.dot(theta)).T).dot(y) - np.log(1 - sigmoid(x.dot(theta)).T).dot(1 - y))/m
    return J

def gradient_function(theta, x, y, m):
    h = sigmoid(x.dot(theta)).reshape(-1, 1)
    y = y.reshape(m, 1)
    gradient = x.T.dot(h - y)/m
    return gradient

print("Initial cost = " + str(cost_function(theta, X, y, size)))
print("Initial gradient = " + str(gradient_function(theta, X, y, size)))

result = opt.fmin_tnc(func=cost_function , x0=theta , fprime=gradient_function , args=(X, y, size))
theta_opt = result[0]
print("Optimal cost = " + str(cost_function(theta_opt, X, y, size)))
theta_opt = theta_opt.reshape((3,1))

linspace = np.linspace(30, 100, 1000)
boundary = -(theta_opt[0] + theta_opt[1]*linspace)/theta_opt[2]
positive = plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], marker='+', c='k')
negative = plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], marker='o', c='b')
plt.plot(linspace, boundary)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend((positive, negative),('Admitted', 'Not admitted'))
plt.show()

def accuracy(theta, x, y, m):
    predictions = sigmoid(x.dot(theta))
    predictions_corrected = [1 if pred >= 0.5 else 0 for pred in predictions]
    number = np.sum(predictions_corrected == y)
    return (float(number)/m)*100

print("Accuracy = " + str(accuracy(theta_opt, X, y, size)) +"%")
