# --------------------------- Lab_work_5 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_5, варіант 6, Група технічних вимог_2:
Кластеризація зображення за кольоровою ознакою з використанням методу k-means.

Package                      Version
---------------------------- -----------
pip                          24.3.1
numpy                        2.2.0
matplotlib                   3.10.0
opencv-python               4.10.0.84
scikit-learn                1.6.0
'''

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------- Функція для завантаження зображення -------------------------
def load_image(image_path):
    '''
    Завантаження зображення та перетворення у формат RGB
    :param image_path: Шлях до зображення
    :return: Масив зображення у форматі RGB
    '''
    image = cv2.imread(image_path)
    if image is None:
        print("Помилка завантаження зображення.")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ------------------- Функція для кластеризації зображення -------------------------
def cluster_image(image, n_clusters):
    '''
    Кластеризація пікселів зображення за кольором з використанням k-means
    :param image: Масив зображення у форматі RGB
    :param n_clusters: Кількість кластерів
    :return: Зображення з кластеризованими кольорами
    '''
    # Перетворення зображення в 2D-масив (пікселі)
    pixels = image.reshape(-1, 3)

    # Застосування k-means кластеризації
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)

    # Замінюємо кольори пікселів на кольори кластерів
    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # Перетворення у вихідний формат зображення
    clustered_image = clustered_pixels.reshape(image.shape).astype(np.uint8)
    return clustered_image

# ------------------- Функція для візуалізації зображень -------------------------
def plot_images(original, clustered, n_clusters):
    '''
    Відображення оригінального та кластеризованого зображення
    :param original: Оригінальне зображення
    :param clustered: Кластеризоване зображення
    :param n_clusters: Кількість кластерів
    '''
    plt.figure(figsize=(10, 5))

    # Оригінальне зображення
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Оригінальне зображення")
    plt.axis('off')

    # Кластеризоване зображення
    plt.subplot(1, 2, 2)
    plt.imshow(clustered)
    plt.title(f"Кластеризація (Кластери = {n_clusters})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ------------------- Головний блок програми -------------------------
if __name__ == '__main__':
    # Шлях до зображення
    image_path = 'lab5.jpg'  # Шлях до вашого зображення

    # Завантаження зображення
    image = load_image(image_path)
    if image is not None:
        # Кількість кластерів
        n_clusters = 5

        # Кластеризація зображення
        clustered_image = cluster_image(image, n_clusters)

        # Відображення результатів
        plot_images(image, clustered_image, n_clusters)

'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Завантаження зображення:
- Оригінальне зображення успішно зчитано та перетворено у формат RGB.

2. Кластеризація зображення:
- Виконано кластеризацію пікселів за кольоровою ознакою методом k-means.
- Використано 5 кластерів для групування пікселів.

3. Візуалізація результатів:
- Оригінальне та кластеризоване зображення успішно відображено.

4. Висновок:
- Скрипт дозволяє автоматизувати кластеризацію кольорів у зображенні.
- Методика k-means підтверджує свою ефективність для аналізу кольорових даних.
'''
