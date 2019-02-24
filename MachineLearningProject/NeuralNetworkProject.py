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

def derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initializeWeights(LIn, LOut):
    weights = np.random.uniform(low=-0.12, high=0.12, size=(LOut, 1 + LIn))
    return weights

def encodeLabels(num_labels, labels):
    labels = np.array(labels)
    oneHot = np.zeros((labels.shape[0], num_labels))
    for i in range(labels.shape[0]):
        oneHot[i][labels[i]-1] = 1
    return oneHot

def costFunction(X, y, theta1, theta2, lambda_reg):
    m = X.shape[0]
    J = (1.0/m) * np.sum(np.sum((-y * np.log(X)) - ((1 - y) * np.log(1 - X))))
    regularization = (np.sum(np.sum(np.square(theta1[:,1:]))) + np.sum(np.sum(np.square(theta2[:,1:])))) * (float(lambda_reg)/(2*m))
    J = J + regularization
    return J

def backprop(params_nn, num_in, num_hid, num_labels, X, y, lambda_reg):
    theta1 = np.reshape(params_nn[:num_hid*(num_in + 1)], (num_hid, (num_in + 1)))
    theta2 = np.reshape(params_nn[num_hid*(num_in + 1):], (num_labels, (num_hid + 1)))
    m = X.shape[0]
    y = encodeLabels(num_labels, y)
    z2 = X.dot(theta1.T)
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    J = costFunction(a3, y, theta1, theta2, lambda_reg)

    theta1Gradient = np.zeros(theta1.shape)
    theta2Gradient = np.zeros(theta2.shape)

    delta3 = a3 - y

    delta2 = (theta2.T.dot(delta3.T)).T * np.hstack((np.ones((z2.shape[0], 1)), derivative(z2)))
    delta2 = delta2[:, 1:]

    theta1Gradient = theta1Gradient + delta2.T.dot(X)
    theta2Gradient = theta2Gradient + delta3.T.dot(a2)

    theta1Gradient = (1/float(m)) * theta1Gradient
    theta2Gradient = (1/float(m)) * theta2Gradient

    theta1Gradient[:, 1:] = theta1Gradient[:, 1:] + (float(lambda_reg)/m)*theta1[:, 1:]
    theta2Gradient[:, 1:] = theta2Gradient[:, 1:] + (float(lambda_reg)/m)*theta2[:, 1:]

    gradients = np.concatenate((theta1Gradient, theta2Gradient), axis=None)

    return J, gradients

def neuralNetwork(X, theta1, theta2):
    a2 = sigmoid(X.dot(theta1.T))
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    a3 = sigmoid(a2.dot(theta2.T))
    predictions = np.zeros((X.shape[0], 1))
    for i in range(len(a3)):
        idx = np.argmax(a3[i])
        idx = idx + 1
        predictions[i] = idx
    return predictions

def accuracy(X, y, theta1, theta2):
    y = y.reshape(len(y), 1)
    predictions = neuralNetwork(X, theta1, theta2)
    number = np.sum(predictions == y)
    accuracy = (float(number)/X.shape[0])*100
    return accuracy

def NNTrainAndTest(XTrain, yTrain, XTest, yTest, numIn, numHid, numOut, lambda_reg):
    thetaTrain1 = initializeWeights(numIn, numHid)
    thetaTrain2 = initializeWeights(numHid, numOut)
    thetasTrain = np.concatenate((thetaTrain1, thetaTrain2), axis=None)
    resultTheta = opt.fmin_tnc(func=backprop, x0=thetasTrain, fprime=None, args=(numIn, numHid, numOut, XTrain, yTrain, lambda_reg), disp=5)
    resultTheta = resultTheta[0]
    theta1 = np.reshape(resultTheta[:numHid*(numIn+1)], ((numHid, numIn+1)))
    theta2 = np.reshape(resultTheta[numHid*(numIn+1):], ((numOut, numHid+1)))

    accuracyTrain = accuracy(XTrain, yTrain, theta1, theta2)
    accuracyTest = accuracy(XTest, yTest, theta1, theta2)
    print("Accuracy train for lambda = " + str(lambda_reg) + " = " + str(accuracyTrain) + "%")
    print("Accuracy test for lambda = " + str(lambda_reg) + " = " + str(accuracyTest) + "%")
    return accuracyTrain, accuracyTest

accuracyTrainValues = []
accuracyTestValues = []
lambdas = np.arange(0.0, 1.6, 0.5)
for lambda_reg in lambdas:
    lambda_reg = round(lambda_reg, 1)
    accuracyTrain, accuracyTest = NNTrainAndTest(XTrain, yTrain, XTest, yTest, 36, 25, 7, lambda_reg)
    accuracyTrainValues.append(accuracyTrain)
    accuracyTestValues.append(accuracyTest)

plt.plot(lambdas, accuracyTrainValues, color='red', label='Train')
plt.plot(lambdas, accuracyTestValues, color='blue', label='Test')
plt.xlabel("Lambda")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
