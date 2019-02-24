from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.svm import SVC

data1 = loadmat('p6/p6/ex6data1.mat')
y1 = data1['y']
X1 = data1['X']

data2 = loadmat('p6/p6/ex6data2.mat')
y2 = data2['y']
X2 = data2['X']

data3 = loadmat('p6/p6/ex6data3.mat')
y3 = data3['y']
X3 = data3['X']
yval3 = data3['yval']
Xval3 = data3['Xval']

for i in range(1, 120, 20):
    svm = SVC(kernel='linear', C=i)
    svm.fit(X1, y1)

    positive = plt.scatter(X1[np.where(y1 == 1), 0], X1[np.where(y1 == 1), 1], marker='+', c='k')
    negative = plt.scatter(X1[np.where(y1 == 0), 0], X1[np.where(y1 == 0), 1], marker='o', c='b')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
    plt.title('SVM with C = ' + str(i))
    plt.show()

svm = SVC(kernel='rbf', C=1, gamma=1 / (2 * 0.1 ** 2))
svm.fit(X2, y2)

positive = plt.scatter(X2[np.where(y2 == 1), 0], X2[np.where(y2 == 1), 1], marker='+', c='k')
negative = plt.scatter(X2[np.where(y2 == 0), 0], X2[np.where(y2 == 0), 1], marker='o', c='b')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
       linestyles=['--', '-', '--'])
plt.title('SVM with gaussian kernel')
plt.show()

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
for C in values:
    for sigma in values:
        svm = SVC(kernel='rbf', C=C, gamma=float(1)/(2 * sigma ** 2))
        svm.fit(X3, y3)
        predictions = svm.predict(Xval3)
        number = 0
        for i in range(len(yval3)):
            if predictions[i] == yval3[i]:
                number = number + 1
        accuracy = (float(number)/len(yval3))*100
        print("Accuracy for C = " + str(C) + " and sigma = " + str(sigma) + " = " + str(accuracy) + "%")

svm = SVC(kernel='rbf', C=1, gamma=float(1)/(2 * 0.1 ** 2))
svm.fit(X3, y3)

positive = plt.scatter(X3[np.where(y3 == 1), 0], X3[np.where(y3 == 1), 1], marker='+', c='k')
negative = plt.scatter(X3[np.where(y3 == 0), 0], X3[np.where(y3 == 0), 1], marker='o', c='b')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
       linestyles=['--', '-', '--'])
plt.title('SVM with best parameters')
plt.show()
