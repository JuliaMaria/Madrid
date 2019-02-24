from scipy.io import loadmat
import numpy as np

data = loadmat('p3/ex3data1.mat')
y = data['y']
X = data['X']

weights = loadmat('p3/ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

X_new = np.ones((5000, 401))
X_new[:, 1:] = X
size = len(y)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def neuralNetwork(X, theta1, theta2):
    a2 = sigmoid(X.dot(theta1.T))
    a2_new = np.ones((a2.shape[0], a2.shape[1]+1))
    a2_new[:, 1:] = a2
    a3 = sigmoid(a2_new.dot(theta2.T))
    predictions = np.zeros(len(a3))
    predictions = predictions.reshape(X.shape[0], 1)
    for i in range(len(a3)):
        idx = np.argmax(a3[i])
        idx = (idx + 1) % 10
        if(idx == 0):
            predictions[i] = 10
        else:
            predictions[i] = idx
    return predictions

predictions = neuralNetwork(X_new, theta1, theta2)

number = np.sum(predictions == y)
accuracy = (float(number)/X.shape[0])*100
print("Accuracy = " + str(accuracy) + "%")
