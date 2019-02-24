from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = loadmat('p3/ex3data1.mat')
y = data['y']
X = data['X']

X_new = np.ones((5000, 401))
X_new[:, 1:] = X
size = len(y)
lambda_reg = 0.1

sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample, :].reshape(-1, 20).T)
plt.axis('off')
plt.show()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cost_function(theta, x, y, lambda_reg):
    m = x.shape[0]
    J = (-np.log(sigmoid(x.dot(theta)).T).dot(y) - np.log(1 - sigmoid(x.dot(theta)).T).dot(1 - y))/m + lambda_reg*np.sum(np.square(theta))/(2*m)
    return J

def gradient_function(theta, x, y, lambda_reg):
    m = x.shape[0]
    h = sigmoid(x.dot(theta)).reshape(-1, 1)
    y = y.reshape(m, 1)
    gradient = np.zeros((theta.shape[0], 1))
    gradient = x.T.dot(h - y)/m
    theta = theta.reshape((theta.shape[0], 1))
    gradient[1:] = gradient[1:] + (lambda_reg/m)*theta[1:]
    return gradient

def oneVsAll(X, y, num_labels, lambda_reg):
    thetas_set = np.zeros((num_labels, 401))
    for i in range(num_labels):
        theta = np.zeros((401, 1))
        example = (y == i + 1) * 1
        result = opt.fmin_tnc(func=cost_function , x0=theta , fprime=gradient_function , args=(X, example, lambda_reg))
        theta_opt = result[0]
        idx = i + 1
        if(i == 9):
            idx = 0
        thetas_set[idx,:] = theta_opt
    return thetas_set

thetas = oneVsAll(X_new, y, 10, lambda_reg)

activations = X_new.dot(thetas.T)
predictions = np.zeros(len(activations))
predictions = predictions.reshape(X.shape[0], 1)

for i in range(len(activations)):
    idx = np.argmax(activations[i])
    if(idx == 0):
        predictions[i] = 10
    else:
        predictions[i] = idx

number = np.sum(predictions == y)
accuracy = (float(number)/X.shape[0])*100
print("Accuracy = " + str(accuracy) + "%")
