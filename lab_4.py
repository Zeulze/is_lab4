import numpy as np
import random
import math

IN_SIZE = 3 #кол-во входных данных
H_SIZE = 12 #кол-во нейронов в первом слое
OUT_SIZE = 3 #выходной вектор

ALPHA = 0.01  # скорость обучения
NUM_EPOCHS = 100 #кол-во эпох

loss_arr = []   # инициализация списка ошибок


def sigm(t):
  return 1 / (1 + math.e ** (-t))

def relu(t):
  return np.maximum(0, t)

def relu_deriv(t): # производная relu(t)
  return (t >= 0).astype(float)

def softmax(t):
  out = np.exp(t)
  return out / np.sum(out)

def sparse_cross_entropy(z, y):
  return -np.log(z[0, y])

def to_full(y, dim):
  y_full = np.zeros((1, dim))
  y_full[0, y] = 1
  return y_full


dataset = [ (np.array([[1, 1, 0]]), 0), (np.array([[1, 0, 1]]), 0), (np.array([[0, 1, 1]]), 1), (np.array([[0, 1, 0]]), 2), (np.array([[0, 1, 1]]), 1), (np.array([[0, 0, 1]]), 2), (np.array([[0, 1, 0]]), 2), (np.array([[1, 1, 1]]), 0), (np.array([[0, 0, 0]]), 1) ]
print(dataset)

W1 = np.random.randn(IN_SIZE, H_SIZE)
b1 = np.random.randn(1, H_SIZE)
W2 = np.random.randn(H_SIZE, OUT_SIZE)
b2 = np.random.randn(1, OUT_SIZE)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/IN_SIZE)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/IN_SIZE)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_SIZE)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_SIZE)


for ep in range(NUM_EPOCHS): #цикл по эпохам
  random.shuffle(dataset) #перемешивание элементов датасета
  for i in range(len(dataset)):#цикл по элементам датасета

    x, y = dataset[i]

    # Прямое распространение
    t1 = x @ W1 + b1 #первый слой
    h1 = relu(t1)#функция активации

    t2 = h1 @ W2 + b2 #второй слой
    z = softmax(t2)#вероятностная мера

    E = sparse_cross_entropy(z, y)#вычисление ошибки

    # Обратное распространение
    y_full = to_full(y, OUT_SIZE) #полный вектор правильного ответа

    dE_dt2 = z - y_full #разница в правильных и неправильных ответах
    dE_dW2 = h1.T @ dE_dt2 #градиент по весам из второго слоя
    dE_db2 = dE_dt2 #градиент по bias из второго слоя
    dE_dh1 = dE_dt2 @ W2.T 
    dE_dt1 = dE_dh1 * relu_deriv(t1) 
    dE_dW1 = x.T @ dE_dt1 #градиент по весам из первого слоя
    dE_db1 = dE_dt1 #градиент по bias из первого слоя

    # Обновление весов
    W1 = W1 - ALPHA * dE_dW1 #изменение весов первого слоя
    b1 = b1 - ALPHA * dE_db1 #изменение bias первого слоя
    W2 = W2 - ALPHA * dE_dW2 #изменение весов второго слоя
    b2 = b2 - ALPHA * dE_db2 #изменение bias второго слоя

    loss_arr.append(E)
    
    
    
def predict(x): #прямое распространение
  t1 = x @ W1 + b1
  h1 = relu(t1)
  t2 = h1 @ W2 + b2
  z = softmax(t2)
  return z #вектор из вероятностей

def calc_accuracy(): #вычисление точности
  correct = 0
  for x, y in dataset:
    z = predict(x)
    y_pred = np.argmax(z)
    if y == y_pred:
      correct += 1
  acc = correct / len(dataset)
  return acc

accuracy = calc_accuracy()
print("Accuracy: ", accuracy)