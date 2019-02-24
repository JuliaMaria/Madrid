import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd

dataTrain = pd.read_csv('datasettrain.csv', header=None, delimiter=' ')
dataTest = pd.read_csv('datasettest.csv', header=None, delimiter=' ')

yTrain = dataTrain.iloc[:, 36]
XTrain = dataTrain.iloc[:, 0:36]
yTest = dataTest.iloc[:, 36]
XTest = dataTest.iloc[:, 0:36]

XTrain.insert(0, -1, 1)
XTest.insert(0, -1, 1)

yTrain = yTrain.values
XTrain = XTrain.values
yTest = yTest.values
XTest = XTest.values

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

def oneVsAll(X, y, num_labels, num_features, lambda_reg):
    thetas_set = np.zeros((num_labels, num_features+1))
    for i in range(0, num_labels):
        theta = np.zeros((num_features+1, 1))
        example = (y == i + 1) * 1
        result = opt.fmin_tnc(func=cost_function , x0=theta , fprime=gradient_function , args=(X, example, lambda_reg), disp=5)
        theta_opt = result[0]
        thetas_set[i,:] = theta_opt
    return thetas_set

def accuracy(X, y, thetas):
    activations = X.dot(thetas.T)
    y = y.reshape(len(y), 1)
    predictions = np.zeros(len(activations))
    predictions = predictions.reshape(X.shape[0], 1)

    for i in range(len(activations)):
        idx = np.argmax(activations[i])
        predictions[i] = idx + 1

    number = np.sum(predictions == y)
    accuracy = (float(number)/X.shape[0])*100
    return accuracy

accuracyTrainValues = []
accuracyTestValues = []
lambdas = np.arange(0.0, 1.6, 0.5)
for lambda_reg in lambdas:
    thetas = oneVsAll(XTrain, yTrain, 7, 36, lambda_reg)
    accuracyTrain = accuracy(XTrain, yTrain, thetas)
    accuracyTest = accuracy(XTest, yTest, thetas)
    accuracyTrainValues.append(accuracyTrain)
    accuracyTestValues.append(accuracyTest)
    print("Accuracy training (lambda = " + str(lambda_reg) + ") = " + str(accuracyTrain) + "%")
    print("Accuracy test (lambda = " + str(lambda_reg) + ") = " + str(accuracyTest) + "%")

plt.plot(lambdas, accuracyTrainValues, color='red', label='Train')
plt.plot(lambdas, accuracyTestValues, color='blue', label='Test')
plt.xlabel("Lambda")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
