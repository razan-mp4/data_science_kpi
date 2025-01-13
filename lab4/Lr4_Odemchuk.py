# --------------------------- Lab_work_4 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_4, варіант 6, ІІ рівень складності:
Розробити програмний скрипт, що реалізує моніторинг змін у асортименті товарів в обраному Інтернет-магазині.

Package                      Version
---------------------------- -----------
pip                          23.1
requests                     2.28.2
beautifulsoup4              4.12.2
pandas                       1.5.3
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# ------------------- Функція для збору даних із сайту -------------------------
def fetch_product_data(url):
    '''
    Збір даних про товари з інтернет-магазину
    :param url: URL сторінки інтернет-магазину
    :return: DataFrame із даними про товари
    '''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Помилка доступу до сайту: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Збір даних про товари для Books to Scrape
    product_names = [item.text.strip() for item in soup.select('.product_pod h3 a')]
    product_prices = [item.text.strip() for item in soup.select('.product_price .price_color')]
    product_availability = [item.text.strip() for item in soup.select('.availability')]

    # Створення DataFrame
    data = {
        'Назва': product_names,
        'Ціна': product_prices,
        'Наявність': product_availability
    }
    return pd.DataFrame(data)

# ------------------- Функція для збереження даних -------------------------
def save_data_to_file(data, file_path):
    '''
    Збереження даних у файл
    :param data: DataFrame із даними про товари
    :param file_path: Шлях до файлу
    '''
    data.to_csv(file_path, index=False)
    print(f"Дані збережено у файл: {file_path}")

# ------------------- Функція для порівняння даних -------------------------
def compare_data(new_data, file_path):
    '''
    Порівняння нових і попередніх даних
    :param new_data: DataFrame із новими даними
    :param file_path: Шлях до файлу зі старими даними
    :return: Зміни у асортименті
    '''
    if not os.path.exists(file_path):
        print("Попередні дані відсутні. Збережено нові дані.")
        save_data_to_file(new_data, file_path)
        return None

    old_data = pd.read_csv(file_path)

    # Порівняння
    added_products = new_data[~new_data['Назва'].isin(old_data['Назва'])]
    removed_products = old_data[~old_data['Назва'].isin(new_data['Назва'])]

    updated_prices = new_data.merge(old_data, on='Назва', suffixes=('_new', '_old'))
    updated_prices = updated_prices[updated_prices['Ціна_new'] != updated_prices['Ціна_old']]

    # Збереження нових даних
    save_data_to_file(new_data, file_path)

    return {
        'Додані товари': added_products,
        'Видалені товари': removed_products,
        'Оновлені ціни': updated_prices
    }

# ------------------- Головний блок програми -------------------------
if __name__ == '__main__':
    # URL сторінки інтернет-магазину
    url = 'http://books.toscrape.com/catalogue/category/books_1/index.html'

    # Шлях до файлу з даними
    file_path = 'product_data.csv'

    # Збір даних
    new_data = fetch_product_data(url)
    if new_data is not None:
        print("Зібрані дані:")
        print(new_data.head())

        # Порівняння даних
        changes = compare_data(new_data, file_path)

        if changes:
            print("Додані товари:")
            print(changes['Додані товари'])

            print("Видалені товари:")
            print(changes['Видалені товари'])

            print("Оновлені ціни:")
            print(changes['Оновлені ціни'])

'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Вхідні дані:
- Дані про товари з інтернет-магазину.
- Формат: назва, ціна, наявність.

2. Збір даних:
- Використано бібліотеки requests і BeautifulSoup для парсингу HTML сторінки.
- Дані збережено у форматі CSV.

3. Порівняння даних:
- Порівняно нові і старі дані для виявлення змін у асортименті.

4. Висновок:
- Розроблений скрипт дозволяє автоматизувати моніторинг змін у асортименті товарів в інтернет-магазині.
'''
