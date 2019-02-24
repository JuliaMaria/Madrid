import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv

data = read_csv('p1/ex1data2.csv', header=None).values.astype(float)
x = data[:,:2]
y = data[:,2]
size = y.size
y.shape = (size, 1)

def normalize(X):
    mu = []
    sigma = []
    X_norm = X
    features = X.shape[1]
    for i in range(features):
        mean = np.mean(X[:, i])
        std_dev = np.std(X[:, i])
        mu.append(mean)
        sigma.append(std_dev)
        X_norm[:, i] = (X_norm[:, i] - mean) / std_dev
    return X_norm, mu, sigma

x, mu, sigma = normalize(x)

X = np.ones((size,3))
X[:,1:3] = x

theta = np.zeros((3, 1))
iterations = 1500
alpha = 0.01

def cost_function(X, y, theta):
    size = y.size
    pred = X.dot(theta)
    sse = (pred - y)
    J = (1.0 / (2 * size)) * sse.T.dot(sse)
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    size = y.size
    Js = np.zeros((iterations+1, 1))
    Js[0, 0] = cost_function(X, y, theta)
    for i in range(iterations):
        pred = X.dot(theta)
        theta_size = theta.size
        for j in range(theta_size):
            xj = X[:, j]
            xj.shape = (size, 1)
            err = (pred - y) * xj
            theta[j][0] = theta[j][0] - alpha * (1.0 / size) * err.sum()
        Js[i+1, 0] = cost_function(X, y, theta)
    return theta, Js

result = gradient_descent(X, y, theta, alpha, iterations)
prediction = np.array([1.0, ((1650.0 - mu[0]) / sigma[0]), ((3 - mu[1]) / sigma[1])]).dot(theta)
print("Prediction: %s" % prediction)

alphas = [round(0.1 * 0.1**i, i+2) for i in range(4)]
print("Gradient descent for different values of alpha")
for alpha in alphas:
    theta = np.zeros((3, 1))
    result1 = gradient_descent(X, y, theta, alpha, iterations)
    theta = np.zeros((3, 1))
    result2 = gradient_descent(X, y, theta, alpha*3, iterations)
    r1 = plt.plot(range(iterations+1), result1[1], label=alpha)
    r2 = plt.plot(range(iterations+1), result2[1], label=alpha*3)
    plt.xlabel("Epochs")
    plt.ylabel("Cost function")
    plt.legend()
    plt.show()
