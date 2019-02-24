import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

data = read_csv('p1/ex1data1.csv', header=None).values.astype(float)
x = data[:,0]
y = data[:,1]
size = len(y)
plt.scatter(x, y)
plt.xlabel("Population")
plt.ylabel("Income")
plt.show()

X = np.ones((size,2))
X[:,1] = x

theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

def cost_function(X, y, theta):
    size = y.size
    pred = X.dot(theta).flatten()
    sse = (pred - y) ** 2
    J = (1.0 / (2 * size)) * sse.sum()
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    size = y.size
    Js = np.zeros((iterations, 1))
    thetas = np.zeros((iterations, 2))

    for i in range(iterations):
        pred = X.dot(theta).flatten()
        err_theta0 = (pred - y) * X[:, 0]
        err_theta1 = (pred - y) * X[:, 1]
        theta[0][0] = theta[0][0] - alpha * (1.0 / size) * err_theta0.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / size) * err_theta1.sum()
        Js[i, 0] = cost_function(X, y, theta)
        thetas[i][0] = theta[0][0]
        thetas[i][1] = theta[1][0]
    return theta, Js, thetas

result = gradient_descent(X, y, theta, alpha, iterations)
plt.scatter(x, y)
plt.xlabel("Population")
plt.ylabel("Income")
plt.show()
linspace = np.linspace(5, 22.5, 1000)
plt.scatter(x, y)
plt.plot(linspace, result[0][0] + linspace*result[0][1], color='red')
plt.xlabel("Population")
plt.ylabel("Income")
plt.show()
plt.plot(range(iterations), result[1])
plt.xlabel("Epochs")
plt.ylabel("Cost function")
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
a = np.arange(-10, 10, 0.05)
b = np.arange(-1, 4, 0.05)
A, B = np.meshgrid(a, b)
zs = np.array([cost_function(X, y, np.vstack((t0, t1))) for t0,t1 in zip(np.ravel(A), np.ravel(B))])
Z = zs.reshape(A.shape)

ax.plot_surface(A, B, Z)
ax.set_xlabel("Theta0")
ax.set_ylabel("Theta1")
ax.set_zlabel("Cost function")
plt.show()

theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-1, 4, 100)
J = np.zeros((theta0.size, theta1.size))

for num0, th0 in enumerate(theta0):
    for num1, th1 in enumerate(theta1):
        J[num0, num1] = cost_function(X, y, np.vstack((th0, th1)))

J = J.T
plt.contour(theta0, theta1, J, np.logspace(-2, 3, 20))
for i in range(iterations):
    if i%100 == 0:
        plt.scatter(result[2][i][0], result[2][i][1], color ='blue')
plt.scatter(result[0][0], result[0][1], color='red', marker='x')
plt.xlabel("Theta0")
plt.ylabel("Theta1")
plt.show()
