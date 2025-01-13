# -------- Залік  --------
'''
Виконав: Одемчук Назар
Варіант: №11
Завдання:
1. Ідентифікувати, чи є на зображенні автомобіль.
2. Використати попередньо навчену модель MobileNetV2.
3. Вивести список найімовірніших передбачень з їх ймовірністю та повідомити результат.

Використані бібліотеки:
- Python: 3.10
- TensorFlow: 2.10.0
- NumPy: 1.23.5
- Pillow: 9.3.0
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Функція для завантаження та підготовки зображення
def prepare_image(image_path, target_size=(224, 224)):
    """
    Завантажує та підготовлює зображення для передбачення.
    :param image_path: Шлях до зображення
    :param target_size: Розмір для масштабування
    :return: Попередньо оброблене зображення
    """
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Основна функція для ідентифікації автомобіля
def identify_vehicle(image_path):
    """
    Ідентифікує, чи є на зображенні автомобіль.
    :param image_path: Шлях до зображення
    :return: Результат передбачення
    """
    # Завантаження моделі MobileNetV2
    model = MobileNetV2(weights="imagenet")

    # Підготовка зображення
    prepared_image = prepare_image(image_path)

    # Передбачення
    predictions = model.predict(prepared_image)

    # Розшифровка результатів
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    # Перевірка наявності автомобіля серед передбачень
    for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({confidence * 100:.2f}%)")
        if any(word in label.lower() for word in ["car", "vehicle", "minivan", "bus", "jeep", "wagon", "cabcar"]):
            return f"На зображенні знайдено автомобіль: {label} з ймовірністю {confidence * 100:.2f}%"

    return "На зображенні автомобіль не знайдено."


# Основний блок програми
if __name__ == "__main__":
    # Введіть шлях до зображення
    image_path = input("Введіть шлях до зображення: ")

    # Ідентифікація автомобіля
    result = identify_vehicle(image_path)

    # Виведення результату
    print(result)

"""
Очікувані результати:
1. Користувач вводить шлях до зображення (наприклад, "car.jpg").
2. Виводиться список найвірогідніших передбачень з їх ймовірністю.
3. Якщо серед передбачень є автомобіль, програма повідомляє про це.
"""
