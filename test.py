import numpy as np
from keras.datasets import mnist
import os

(train_X, train_y), (test_X, test_y) = mnist.load_data()


# Функция активации ReLU
def relu(x):
    return np.maximum(0, x)


def forward_pass(x, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(x, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    return Z1, A1, Z2, A2, Z3, A3


# функция для преобразования массива 728 в массив 28 на 28
def split(arr):
    arr = arr[0]
    arrs = []
    for i in range(28):
        a = []
        for j in range(28):
            a.append(arr[28 * i + j])
        arrs.append(a)
    return arrs


def test(X, y, W1, b1, W2, b2, W3, b3):
    from matplotlib import pyplot as plt
    for x, target in zip(X, y):
        x = x.reshape(1, -1)
        Z1, A1, Z2, A2, Z3, A3 = forward_pass(x, W1, b1, W2, b2, W3, b3)
        x1 = split(x)
        plt.imshow(x1)
        plt.title(f'Верное значение: {target} - ответ нейросети = {np.argmax(A3, axis=1)}')
        plt.show()


if os.path.exists("test_2size_2.npz"):
    a = np.load("test_2size_2.npz")
    W1, b1, W2, b2, W3, b3 = a["A"], a['B'], a['C'], a["D"], a["E"], a["F"]


X_test = test_X.reshape(len(test_X), -1) / 255.0
y_test = test_y

test(X_test, y_test, W1, b1, W2, b2, W3, b3)