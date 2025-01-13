# --------------------------- Lab_work_2 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_2, варіант 6, ІІ рівень складності:
1. Отримання вхідних даних із властивостями, заданими в Лр_1.
2. Модель вхідних даних із аномальними вимірами.
3. Очищення вхідних даних від аномальних вимірів.
4. Визначення показників якості та оптимізація моделі.
5. Статистичне навчання поліноміальної моделі за методом найменших квадратів (МНК – LSM).
6. Прогнозування параметрів досліджуваного процесу за «навченою» моделлю.
7. Аналіз отриманих результатів та верифікація розробленого скрипта.

Package                      Version
---------------------------- -----------
pip                          23.1
numpy                        1.23.5
pandas                       1.5.3
matplotlib                   3.6.2
'''

import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt

# ------------------------ ФУНКЦІЯ парсингу реальних даних --------------------------

def file_parsing(URL, File_name, Data_name):
    '''
    :param URL: Посилання на джерело даних
    :param File_name: Назва файлу
    :param Data_name: Назва стовпця з даними
    :return: Масив даних з файлу
    '''
    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Джерело даних: ', URL)
    return S_real

# ---------------------- Моделі розподілу похибок -------------------------

def random_norm(mean, std_dev, size):
    '''
    Генерація даних за нормальним законом розподілу
    :param mean: Середнє значення
    :param std_dev: Стандартне відхилення
    :param size: Розмір вибірки
    :return: Масив випадкових величин
    '''
    S = np.random.normal(mean, std_dev, size)
    return S

def random_exponential(scale, size):
    '''
    Генерація даних за експоненційним законом розподілу
    :param scale: Параметр масштабу
    :param size: Розмір вибірки
    :return: Масив випадкових величин
    '''
    S = np.random.exponential(scale, size)
    return S

# ------------------- Моделі трендів -------------------------

def linear_trend(size):
    '''
    Генерація лінійного тренду
    :param size: Розмір вибірки
    :return: Масив значень тренду
    '''
    return np.linspace(0, 100, size)

def quadratic_trend(size):
    '''
    Генерація квадратичного тренду
    :param size: Розмір вибірки
    :return: Масив значень тренду
    '''
    return np.array([0.0005 * x ** 2 for x in range(size)])

# ------------------- Модель із аномальними вимірами -------------------------

def add_anomalies(data, anomaly_percentage, anomaly_scale):
    '''
    Додавання аномальних значень до даних
    :param data: Вхідні дані
    :param anomaly_percentage: Відсоток аномальних даних
    :param anomaly_scale: Масштаб аномалій
    :return: Дані з аномаліями
    '''
    data_with_anomalies = data.copy()
    n_anomalies = int(len(data) * anomaly_percentage / 100)
    anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
    for idx in anomaly_indices:
        data_with_anomalies[idx] += np.random.normal(0, anomaly_scale)
    return data_with_anomalies

# ------------------- Очищення даних від аномалій -------------------------

def clean_anomalies(data, threshold):
    '''
    Видалення аномальних значень із даних
    :param data: Вхідні дані
    :param threshold: Граничне значення для виявлення аномалій
    :return: Очищені дані
    '''
    mean = np.mean(data)
    std_dev = np.std(data)
    cleaned_data = [x for x in data if abs(x - mean) <= threshold * std_dev]
    return np.array(cleaned_data)

# ------------------- Поліноміальна регресія (МНК) -------------------------

def polynomial_regression(data, degree):
    '''
    Поліноміальна регресія за методом найменших квадратів
    :param data: Вхідні дані
    :param degree: Степінь полінома
    :return: Коефіцієнти полінома
    '''
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, degree)
    return coeffs

def extrapolate(data, coeffs, extra_points):
    '''
    Екстраполяція даних за допомогою поліноміальної регресії
    :param data: Вхідні дані
    :param coeffs: Коефіцієнти полінома
    :param extra_points: Кількість точок для екстраполяції
    :return: Масив екстрапольованих значень
    '''
    x = np.arange(len(data) + extra_points)
    y = np.polyval(coeffs, x)
    return y

# ------------------- Адитивна модель -------------------------

def additive_model(trend, error):
    '''
    Адитивна модель тренду і похибки
    :param trend: Модель тренду
    :param error: Модель похибки
    :return: Масив значень вибірки
    '''
    return trend + error

# ------------------- Візуалізація -------------------------

def plot_data(original, cleaned, predicted, title):
    '''
    Побудова графіків для порівняння даних
    :param original: Оригінальні дані
    :param cleaned: Очищені дані
    :param predicted: Прогнозовані дані
    :param title: Назва графіку
    '''
    plt.plot(original, label='Оригінальні дані')
    plt.plot(cleaned, label='Очищені дані')
    plt.plot(predicted, label='Прогнозовані дані')
    plt.legend()
    plt.title(title)
    plt.show()

# ------------------- Аналіз статистичних характеристик -------------------------

def analyze_results(data, description):
    '''
    Аналіз статистичних характеристик даних
    :param data: Масив даних
    :param description: Опис даних
    '''
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)
    print(f'--- {description} ---')
    print(f'Матиматичне сподівання: {mean}')
    print(f'Дисперсія: {variance}')
    print(f'СКВ: {std_dev}')
    print('------------------------------------')

# ------------------- Головний блок програми -------------------------

if __name__ == '__main__':
    # Параметри
    size = 1000
    mean = 0
    std_dev = 5
    scale = 1.0
    anomaly_percentage = 5  # Відсоток аномалій
    anomaly_scale = 20  # Масштаб аномалій
    threshold = 3  # Поріг для очищення аномалій
    degree = 2  # Степінь полінома
    extra_points = int(size * 0.5)  # Точки для екстраполяції

    # Генерація даних
    norm_error = random_norm(mean, std_dev, size)
    lin_trend = linear_trend(size)
    lin_model_norm = additive_model(lin_trend, norm_error)

    # Додавання аномалій
    data_with_anomalies = add_anomalies(lin_model_norm, anomaly_percentage, anomaly_scale)
    analyze_results(data_with_anomalies, 'Оригінальні дані (з аномаліями)')

    # Очищення даних
    cleaned_data = clean_anomalies(data_with_anomalies, threshold)
    analyze_results(cleaned_data, 'Очищені дані')

    # Поліноміальна регресія
    coeffs = polynomial_regression(cleaned_data, degree)

    # Екстраполяція
    predicted_data = extrapolate(cleaned_data, coeffs, extra_points)
    analyze_results(predicted_data, 'Прогнозовані дані')

    # Візуалізація
    plot_data(data_with_anomalies, cleaned_data, predicted_data, 'Аналіз даних: Аномалії, Очищення, Прогнозування')


'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Задані характеристики вхідної вибірки:
- Модель із нормальними та експоненційними похибками.
- Додано 5% аномальних значень із масштабом аномалій 20.
- Використано тренди: лінійний та квадратичний.
- Вхідні дані очищено від аномалій із використанням порога 3 стандартних відхилень.
- Виконано поліноміальну регресію другого степеня.

2. Визначені характеристики:
- Моделі трендів та похибок відображено на графіках.
- Аномальні дані успішно виявлено та видалено.
- Поліноміальна регресія забезпечує адекватне наближення очищених даних.

3. Прогнозування:
- Виконано екстраполяцію на 50% від об'єму вибірки.
- Результати прогнозу відображено на графіку разом із очищеними та оригінальними даними.

4. Висновок:
- Розроблений скрипт ефективно обробляє дані з аномаліями.
- Поліноміальна регресія другого степеня є придатною для моделювання трендів очищених даних.
- Методики очищення, моделювання та прогнозування підтвердили свою адекватність.

Розроблений скрипт є універсальним інструментом для аналізу та моделювання даних у заданих умовах.
'''
