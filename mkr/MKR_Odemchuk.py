# -------- MKR Variant 6 --------
'''
Виконав: Одемчук Назар
Варіант: №6
Завдання:
1. Реалізувати скрипт із згладжуванням вибірки вимірів за методом найменших квадратів.
2. Побудувати графік вихідних даних та згладженої моделі.
3. Результати вивести у вигляді графіка та коефіцієнтів моделі.

Використані бібліотеки:
- Python: 3.10
- NumPy: 1.23.5
- Matplotlib: 3.6.2
'''

import numpy as np
import matplotlib.pyplot as plt

# Функція для генерації вибірки даних
def generate_sample(num_points=50, noise_level=2):
    """
    Генерує вибірку даних із шумом.
    :param num_points: Кількість точок вибірки
    :param noise_level: Рівень шуму
    :return: x, y (незалежна та залежна змінні)
    """
    np.random.seed(42)
    x = np.linspace(0, 10, num_points)
    y = 2.5 * x + np.random.normal(0, noise_level, size=num_points)  # Лінійна залежність із шумом
    return x, y

# Функція для розрахунку коефіцієнтів згладжувальної лінії
def least_squares_fit(x, y):
    """
    Розрахунок коефіцієнтів методом найменших квадратів.
    :param x: Незалежна змінна
    :param y: Залежна змінна
    :return: Коефіцієнти a0 (вільний член) та a1 (нахил)
    """
    A = np.vstack([x, np.ones(len(x))]).T
    coefficients, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coefficients

# Функція для побудови графіків
def plot_results(x, y, y_pred):
    """
    Візуалізація результатів згладжування.
    :param x: Незалежна змінна
    :param y: Залежна змінна (вихідна вибірка)
    :param y_pred: Прогнозовані значення
    """
    plt.scatter(x, y, label='Вихідні дані', color='blue')
    plt.plot(x, y_pred, color='red', label='Лінія згладжування')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Згладжування за методом найменших квадратів')
    plt.legend()
    plt.grid()
    plt.show()

# Основний блок програми
if __name__ == "__main__":
    # Генерація вибірки
    x, y = generate_sample(num_points=50, noise_level=2)

    # Розрахунок коефіцієнтів
    coefficients = least_squares_fit(x, y)
    a1, a0 = coefficients[0], coefficients[1]

    # Побудова прогнозу
    y_pred = a1 * x + a0

    # Виведення коефіцієнтів
    print(f"Коефіцієнти моделі: a0 = {a0:.2f}, a1 = {a1:.2f}")

    # Побудова графіків
    plot_results(x, y, y_pred)

'''
Очікувані результати:
1. Графік:
    - Сині точки — вихідні дані (з шумом).
    - Червона лінія — згладжувальна модель.
2. Вивід у консолі:
    - Коефіцієнти моделі, наприклад:
    Коефіцієнти моделі: a0 = -0.11, a1 = 2.52
'''
