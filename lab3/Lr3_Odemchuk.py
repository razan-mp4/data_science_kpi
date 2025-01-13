# --------------------------- Lab_work_3 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_3, варіант 6, І рівень складності:
1. Розробка програмного скрипта для оцінювання ефективності впровадження нового товару.
2. Використання 12 критеріїв, 7 з яких максимізовані, а решта – мінімізовані.
3. Кількість аналогічних товарів – 8.
4. Вхідні дані містяться у файлі. Формат, зміст показників та критеріїв обрано самостійно.

Package                      Version
---------------------------- -----------
pip                          23.1
numpy                        1.23.5
pandas                       1.5.3
matplotlib                   3.6.2
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- Генерація даних для файлу -------------------------
def generate_sample_data(file_path):
    '''
    Генерація тестових даних і збереження у файл
    :param file_path: Шлях до файлу
    '''
    np.random.seed(42)
    data = {
        'Criterion_1': np.random.randint(50, 100, 8),
        'Criterion_2': np.random.randint(50, 100, 8),
        'Criterion_3': np.random.randint(50, 100, 8),
        'Criterion_4': np.random.randint(50, 100, 8),
        'Criterion_5': np.random.randint(50, 100, 8),
        'Criterion_6': np.random.randint(50, 100, 8),
        'Criterion_7': np.random.randint(50, 100, 8),
        'Criterion_8': np.random.randint(10, 50, 8),
        'Criterion_9': np.random.randint(10, 50, 8),
        'Criterion_10': np.random.randint(10, 50, 8),
        'Criterion_11': np.random.randint(10, 50, 8),
        'Criterion_12': np.random.randint(10, 50, 8),
    }
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"Тестові дані збережено у файл: {file_path}")

# ------------------- Зчитування даних із файлу -------------------------
def read_data_from_file(file_path):
    '''
    Зчитування даних із Excel або CSV файлу
    :param file_path: Шлях до файлу
    :return: DataFrame із вхідними даними
    '''
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Формат файлу має бути CSV або Excel")
        print("Дані успішно зчитано із файлу")
        return data
    except Exception as e:
        print(f"Помилка зчитування файлу: {e}")
        return None

# ------------------- Нормалізація даних -------------------------
def normalize_data(data, criteria_types):
    '''
    Нормалізація даних залежно від типу критерія (максимізація/мінімізація)
    :param data: DataFrame із вхідними даними
    :param criteria_types: Список типів критеріїв ('max' або 'min')
    :return: Нормалізований DataFrame
    '''
    normalized_data = data.copy()
    for i, crit_type in enumerate(criteria_types):
        if crit_type == 'max':
            normalized_data.iloc[:, i] = data.iloc[:, i] / data.iloc[:, i].max()
        elif crit_type == 'min':
            normalized_data.iloc[:, i] = data.iloc[:, i].min() / data.iloc[:, i]
    print("Дані нормалізовано залежно від типу критеріїв")
    return normalized_data

# ------------------- Розрахунок ефективності -------------------------
def calculate_efficiency(normalized_data):
    '''
    Розрахунок ефективності для кожного товару
    :param normalized_data: Нормалізований DataFrame із даними
    :return: Серія значень ефективності
    '''
    efficiency = normalized_data.mean(axis=1)
    print("Ефективність розраховано")
    return efficiency

# ------------------- Візуалізація результатів -------------------------
def plot_efficiency(efficiency):
    '''
    Побудова графіку ефективності
    :param efficiency: Серія значень ефективності
    '''
    plt.bar(range(len(efficiency)), efficiency, color='skyblue')
    plt.xlabel('Товари')
    plt.ylabel('Ефективність')
    plt.title('Ефективність впровадження нового товару')
    plt.show()

# ------------------- Головний блок програми -------------------------
if __name__ == '__main__':
    # Шлях до файлу з даними
    file_path = 'products_data.xlsx'

    # Генерація тестових даних
    generate_sample_data(file_path)

    # Типи критеріїв ('max' для максимізації, 'min' для мінімізації)
    criteria_types = ['max', 'max', 'max', 'max', 'max', 'max', 'max', 'min', 'min', 'min', 'min', 'min']

    # Зчитування даних
    data = read_data_from_file(file_path)
    if data is not None:
        print("Вхідні дані:")
        print(data.head())

        # Нормалізація даних
        normalized_data = normalize_data(data, criteria_types)

        # Розрахунок ефективності
        efficiency = calculate_efficiency(normalized_data)

        # Виведення результатів
        print("Ефективність:")
        print(efficiency)

        # Візуалізація результатів
        plot_efficiency(efficiency)

'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Вхідні дані:
- Містять 12 критеріїв (7 максимізованих, 5 мінімізованих).
- Формат файлу: Excel із даними для 8 аналогічних товарів.

2. Нормалізація даних:
- Для максимізованих критеріїв використовувалась нормалізація за максимальним значенням.
- Для мінімізованих критеріїв нормалізація виконана за мінімальним значенням.

3. Розрахунок ефективності:
- Для кожного товару розраховано середнє значення нормалізованих критеріїв.

4. Візуалізація результатів:
- Побудовано стовпчастий графік, що відображає ефективність кожного товару.

5. Висновок:
- Розроблений скрипт ефективно виконує оцінювання впровадження нового товару за заданими критеріями.
'''
