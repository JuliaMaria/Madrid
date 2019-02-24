import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.svm import SVC
import pandas as pd

dataTrain = pd.read_csv('datasettrain.csv', header=None, delimiter=' ')
dataTest = pd.read_csv('datasettest.csv', header=None, delimiter=' ')

yTrain = dataTrain.iloc[:, 36]
XTrain = dataTrain.iloc[:, 0:36]
yTest = dataTest.iloc[:, 36]
XTest = dataTest.iloc[:, 0:36]

yTrain = yTrain.values
XTrain = XTrain.values
yTest = yTest.values
XTest = XTest.values

def accuracy(svm, X, y):
    predictions = svm.predict(X)
    number = 0
    for i in range(len(y)):
        if predictions[i] == y[i]:
            number = number + 1
    accuracy = (float(number)/len(y))*100
    return accuracy

def SVMTrainAndTest(svm, XTrain, yTrain, XTest, yTest):
    svm.fit(XTrain, yTrain)
    accuracyTrain = accuracy(svm, XTrain, yTrain)
    accuracyTest = accuracy(svm, XTest, yTest)
    return accuracyTrain, accuracyTest

accuracyTrainLinear = []
accuracyTestLinear = []

CValues = [0.01, 0.1, 1]
SigmaValues = [0.01, 0.1, 1, 10, 100]

def SVMParameterSearch(XTrain, yTrain, XTest, yTest, CValues, SigmaValues):
    for C in CValues:
        svmLin = SVC(kernel='linear', C=C)
        accuracyTrain, accuracyTest = SVMTrainAndTest(svmLin, XTrain, yTrain, XTest, yTest)
        accuracyTrainLinear.append(accuracyTrain)
        accuracyTestLinear.append(accuracyTest)
        print("Train accuracy for SVM with linear kernel for C = " + str(C) + " = " + str(accuracyTrain) + "%")
        print("Test accuracy for SVM with linear kernel for C = " + str(C) + " = " + str(accuracyTest) + "%")
        for sigma in SigmaValues:
            svmRBF = SVC(kernel='rbf', C=C, gamma=1.0/(2 * sigma ** 2))
            accuracyTrain, accuracyTest = SVMTrainAndTest(svmRBF, XTrain, yTrain, XTest, yTest)
            print("Train accuracy for SVM with RBF kernel for C = " + str(C) + " and sigma = " + str(sigma) + " = " + str(accuracyTrain) + "%")
            print("Test accuracy for SVM with RBF kernel for C = " + str(C) + " and sigma = " + str(sigma) + " = " + str(accuracyTest) + "%")

SVMParameterSearch(XTrain, yTrain, XTest, yTest, CValues, SigmaValues)

plt.plot(CValues, accuracyTrainLinear, color='red', label='Train')
plt.plot(CValues, accuracyTestLinear, color='blue', label='Test')
plt.xlabel("C")
plt.ylabel("Accuracy for linear kernel (%)")
plt.legend()
plt.show()
