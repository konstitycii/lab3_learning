# Импорт библиотек
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Модифицированный класс MLPtorch с тремя скрытыми слоями
class MLPptorch(nn.Module):
    def __init__(self, in_size, first_hidden_size, second_hidden_size, third_hidden_size, out_size):
        super(MLPptorch, self).__init__()
        # Определение слоев нейронной сети
        self.layers = nn.Sequential(
            nn.Linear(in_size, first_hidden_size), nn.ReLU(),
            nn.Linear(first_hidden_size, second_hidden_size), nn.ReLU(),
            nn.Linear(second_hidden_size, third_hidden_size), nn.ReLU(),
            nn.Linear(third_hidden_size, out_size), nn.Sigmoid()
        )

    # Прямой проход
    def forward(self, x):
        return self.layers(x)

# Функция обучения модели
def train_model(x, y, num_iterations, net, loss_fn, optimizer):
    for i in range(num_iterations):
        # Прямой проход
        pred = net(x)
        # Рассчет функции потерь
        loss = loss_fn(pred, y)
        # Обратное распространение ошибки и обновление весов
        loss.backward()
        optimizer.step()
        # Вывод ошибки на экран каждые PRINT_INTERVAL итераций
        if i % PRINT_INTERVAL == 0:
            print(f'Ошибка на {i} итерации: {loss.item()}')
    return loss.item()

# Загрузка данных
df = pd.read_csv('/home/and/python_poned/lab3/data.csv')
# Перемешивание данных
df = df.iloc[np.random.permutation(len(df))]

# Подготовка данных для обучения
X = df.iloc[0:100, 0:3].values
y = df.iloc[0:100, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y[:, i - 1] = np.where(y == i, 1, 0).reshape(1, -1)

# Подготовка данных для тестирования
X_test = df.iloc[100:150, 0:3].values
y = df.iloc[100:150, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
Y_test = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y_test[:, i - 1] = np.where(y == i, 1, 0).reshape(1, -1)

# Определение размеров входа и выхода
input_size = X.shape[1]
first_hidden_size = 50
second_hidden_size = 30
third_hidden_size = 20
output_size = Y.shape[1] if len(Y.shape) > 1 else 1

# Создание модели
net = MLPptorch(input_size, first_hidden_size, second_hidden_size, third_hidden_size, output_size)
# Выбор функции потерь (Mean Squared Error)
loss_fn = nn.MSELoss()
# Выбор оптимизатора (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(net.parameters(), lr=0.009)

# Константы
PRINT_INTERVAL = 200
LEARNING_RATE = 0.009
NUM_ITERATIONS = 5000

# Обучение модели
train_model(torch.Tensor(X.astype(np.float32)), torch.Tensor(Y.astype(np.float32)), NUM_ITERATIONS, net, loss_fn, optimizer)

# Предсказание на тренировочных данных
pred_train = net(torch.Tensor(X.astype(np.float32))).detach().numpy()
# Вычисление ошибки на тренировочных данных
err_train = sum(abs((pred_train > 0.5) - Y))
print(err_train)

# Предсказание на тестовых данных
pred_test = net(torch.Tensor(X_test.astype(np.float32))).detach().numpy()
# Вычисление ошибки на тестовых данных
err_test = sum(abs((pred_test > 0.5) - Y_test))
print(err_test)