import numpy as np
from keras.datasets import mnist
import os

(train_X, train_y), (test_X, test_y) = mnist.load_data()


# Функция активации ReLU и ее производная
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# прямой ход (получение предположительного ответа нейронной сети)
def forward_pass(x, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(x, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)

    return Z1, A1, Z2, A2, Z3, A3


# функция обратного распространиения ошибки
def backward_pass(x, y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, b1, b2, b3, learning_rate):
    # получение разницы между ожиданием и полученным ответом сети
    output_delta = A3 - y
    # получение ошибки для 2 слоя весов
    hidden_error_2 = output_delta.dot(W3.T)
    hidden_delta_2 = hidden_error_2 * relu_derivative(A2)
    # получение ошибки для 1 слоя весов
    hidden_error_1 = hidden_delta_2.dot(W2.T)
    hidden_delta_1 = hidden_error_1 * relu_derivative(A1)

    # редактируем веса, которые идут к выходноу слою
    W3 -= np.dot(A2.T, output_delta) * (learning_rate)
    b3 -= np.sum(output_delta, axis=0, keepdims=True) * (learning_rate)
    # редактируем веса для втрого слоя нейронов
    W2 -= np.dot(A1.T, hidden_delta_2) * (learning_rate)
    b2 -= np.sum(hidden_delta_2, axis=0, keepdims=True) * (learning_rate)
    # редактируем веса для первого слоя нейронов
    W1 -= np.outer(x, hidden_delta_1) * (learning_rate)
    b1 -= np.sum(hidden_delta_1, axis=0, keepdims=True) * (learning_rate)


# Инициализация весов и смещений
input_size = 784
hidden_size_1 = 40
hidden_size_2 = 20
output_size = 10

# инициализация связей или получение их из файла
if os.path.exists("test_2size_2.npz"):
    a = np.load("test_2size_2.npz")
    W1, b1, W2, b2, W3, b3 = a["A"], a['B'], a['C'], a["D"], a["E"], a["F"]
else:
    np.random.seed(0)
    W1 = np.random.randn(input_size, hidden_size_1) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size_1))
    W2 = np.random.randn(hidden_size_1, hidden_size_2) * np.sqrt(2.0 / hidden_size_1)
    b2 = np.zeros((1, hidden_size_2))
    W3 = np.random.randn(hidden_size_2, output_size) * np.sqrt(2.0 / hidden_size_2)
    b3 = np.zeros((1, output_size))


# получение ответа нейроной сети для всех тестовых значений
def predict(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    return np.argmax(A3, axis=1)


# фукция для тренировки нернной сети
def train(X, y, learning_rate=0.01, epochs=10):
    for epoch in range(epochs):
        print("Epoch:", epoch)
        for x, target in zip(X, y):
            x = x.reshape(1, -1)
            Z1, A1, Z2, A2, Z3, A3 = forward_pass(x, W1, b1, W2, b2, W3, b3)
            backward_pass(x, target, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, b1, b2, b3, learning_rate)
    return W1, b1, W2, b2, W3, b3


# Преобразуем изображения в вектора и нормализуем
X_train = train_X.reshape(len(train_X), -1) / 255.0
# преобразование числа в массив (2 = [0, 0, 1])
# так мы получаем массив с массивами из 10 элементов, так чтобы удобно было сравнивать с ответами нейронной сети
y_train = np.eye(10)[train_y]  # Преобразуем метки классов в one-hot кодировку
X_test = test_X.reshape(len(test_X), -1) / 255.0
y_test = test_y

predictions = predict(X_test, W1, b1, W2, b2, W3, b3)
accuracy = np.mean(predictions == y_test)
print("Точность на тестовых данных до:", accuracy)

W1, b1, W2, b2, W3, b3 = train(X_train, y_train, learning_rate=0.001, epochs=10)

predictions = predict(X_test, W1, b1, W2, b2, W3, b3)
accuracy = np.mean(predictions == y_test)
print("Точность на тестовых данных после:", accuracy)

np.savez("test_2size_2", A=W1, B=b1, C=W2, D=b2, E=W3, F=b3)
