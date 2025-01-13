# --------------------------- Lab_work_6 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_6, варіант 6, І рівень складності:
Розробити програмний скрипт, що синтезує штучну нейронну мережу з
використанням скалярних та матричних операцій та реалізувати її навчання за Data Set:

Вхід:
[[0, 1, 1],
[1, 1, 0],
[1, 0, 0],
[1, 1, 1],
[0, 0, 1]]

Вихід:
[[0],
[0],
[1],
[1],
[1]]

Довести працездатність мережі.

Package                      Version
---------------------------- -----------
numpy                        2.2.0
matplotlib                   3.10.0
'''

import numpy as np
import matplotlib.pyplot as plt

# ------------------- Функція активації (сигмоїда) -------------------------
def sigmoid(x):
    '''
    Функція активації - сигмоїда
    :param x: Значення
    :return: Значення після застосування сигмоїди
    '''
    return 1 / (1 + np.exp(-x))

# ------------------- Похідна сигмоїди -------------------------
def sigmoid_derivative(x):
    '''
    Похідна сигмоїди
    :param x: Значення
    :return: Похідна
    '''
    return x * (1 - x)

# ------------------- Функція для навчання нейронної мережі -------------------------
def train_neural_network(inputs, outputs, epochs=10000, learning_rate=0.1):
    '''
    Навчання нейронної мережі з одним прихованим шаром
    :param inputs: Матриця вхідних даних
    :param outputs: Матриця вихідних даних
    :param epochs: Кількість ітерацій навчання
    :param learning_rate: Швидкість навчання
    :return: Ваги, зміщення, прогнозовані виходи, історія втрат
    '''
    np.random.seed(42)

    # Ініціалізація ваг випадковими числами
    weights = np.random.rand(inputs.shape[1], 1)
    bias = np.random.rand(1)

    loss_history = []

    for epoch in range(epochs):
        # Пряме проходження
        linear_output = np.dot(inputs, weights) + bias
        predicted_output = sigmoid(linear_output)

        # Обчислення помилки
        error = outputs - predicted_output
        loss = np.mean(error**2)
        loss_history.append(loss)

        # Оновлення ваг
        d_weights = np.dot(inputs.T, error * sigmoid_derivative(predicted_output))
        weights += learning_rate * d_weights
        bias += learning_rate * np.sum(error * sigmoid_derivative(predicted_output))

    return weights, bias, predicted_output, loss_history

# ------------------- Функція для візуалізації втрат -------------------------
def plot_loss(loss_history):
    '''
    Візуалізація графіка втрат
    :param loss_history: Історія втрат
    '''
    plt.figure()
    plt.plot(loss_history)
    plt.title("Графік втрат під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("Втрати")
    plt.grid()
    plt.show()

# ------------------- Функція для візуалізації результатів -------------------------
def plot_results(outputs, predicted_output):
    '''
    Візуалізація порівняння реальних і прогнозованих виходів
    :param outputs: Реальні виходи
    :param predicted_output: Прогнозовані виходи
    '''
    plt.figure()
    plt.plot(outputs, label="Реальні виходи", marker='o')
    plt.plot(predicted_output, label="Прогнозовані виходи", marker='x')
    plt.title("Порівняння реальних і прогнозованих виходів")
    plt.xlabel("Зразки")
    plt.ylabel("Значення")
    plt.legend()
    plt.grid()
    plt.show()

# ------------------- Головний блок програми -------------------------
if __name__ == '__main__':
    # Вхідні дані
    inputs = np.array([[0, 1, 1],
                        [1, 1, 0],
                        [1, 0, 0],
                        [1, 1, 1],
                        [0, 0, 1]])

    # Вихідні дані
    outputs = np.array([[0], [0], [1], [1], [1]])

    # Навчання нейронної мережі
    weights, bias, predicted_output, loss_history = train_neural_network(inputs, outputs)

    # Виведення результатів
    print("Фінальні ваги:")
    print(weights)
    print("\nФінальний зміщення:")
    print(bias)
    print("\nПрогнозовані виходи:")
    print(predicted_output)

    # Візуалізація результатів
    plot_loss(loss_history)
    plot_results(outputs, predicted_output)

'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Вхідні дані:
- Матриця вхідних даних: 5x3
- Матриця вихідних даних: 5x1

2. Ініціалізація:
- Ваги та зміщення ініціалізовано випадковими числами.

3. Навчання:
- Виконано 10,000 ітерацій навчання із швидкістю 0.1.

4. Результати:
- Отримано навчені ваги та зміщення.
- Прогнозовані виходи відповідають заданим виходам із невеликою похибкою.

5. Висновок:
- Мережа успішно синтезована та навченa для виконання задачі класифікації.
'''
