# --------------------------- Lab_work_9 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_9, І рівень складності:
Реалізувати розрахунки та побудову / візуалізацію на цифровій векторній карті
сітку відстаней між пожежними станціями США (див. приклад Лекцій №16).
Кількість пожежних станцій для побудови сітки відстаней обрати самостійно.
Розрахувати середню відстань між пожежними станціями. Провести верифікацію
результатів розрахунку відстаней за одиницями виміру.
'''

import geopandas as gpd
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import matplotlib.pyplot as plt

# --------------------------- Константи ---------------------------
EARTH_RADIUS_M = 6371000  # Радіус Землі в метрах
FILENAME = "Fire_Stations.shp"
NUMBER_OF_STATIONS = 10

# --------------------------- Функції ---------------------------
def load_fire_stations(filename, number_of_stations):
    """
    Завантажує shapefile з інформацією про пожежні станції.

    :param filename: Назва файлу з даними.
    :param number_of_stations: Кількість станцій для завантаження.
    :return: GeoDataFrame з вибраними станціями та всіма станціями.
    """
    fire_stations = gpd.read_file(filename, rows=number_of_stations)
    all_fire_stations = gpd.read_file(filename)
    return fire_stations, all_fire_stations

def calculate_distances(gdf):
    """
    Обчислює середню відстань між станціями за формулою Haversine.

    :param gdf: GeoDataFrame з координатами станцій.
    :return: Середня відстань між станціями в метрах.
    """
    gdf['longitude'] = gdf['geometry'].apply(lambda point: point.x)
    gdf['latitude'] = gdf['geometry'].apply(lambda point: point.y)

    coords_radians = gdf[['latitude', 'longitude']].applymap(radians)
    distances = haversine_distances(coords_radians) * EARTH_RADIUS_M

    num_stations = len(gdf)
    average_distance = distances.sum() / (num_stations * (num_stations - 1))
    return average_distance

def plot_fire_stations(all_stations, selected_stations):
    """
    Візуалізує всі станції та вибрані станції на карті.

    :param all_stations: GeoDataFrame з усіма станціями.
    :param selected_stations: GeoDataFrame з вибраними станціями.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    all_stations.plot(ax=ax, markersize=20, color="blue", marker="o", zorder=1, label="Всі станції")
    selected_stations.plot(ax=ax, markersize=40, color="red", marker="o", zorder=2, label="Вибрані станції")
    plt.legend()
    plt.title("Розташування пожежних станцій")
    plt.xlabel("Довгота")
    plt.ylabel("Широта")
    plt.show()

def main():
    """
    Головна функція виконання програми.
    """
    # Завантаження даних
    fire_stations, all_fire_stations = load_fire_stations(FILENAME, NUMBER_OF_STATIONS)

    # Візуалізація вибраних станцій
    plot_fire_stations(all_fire_stations, fire_stations)

    # Обчислення середньої відстані
    average_distance = calculate_distances(fire_stations)

    # Виведення результату
    print(f"Середня відстань між пожежними станціями: {average_distance:.2f} метрів")

# --------------------------- Виконання ---------------------------
if __name__ == "__main__":
    main()

'''
Під час виконання лабораторної роботи №9 було виконано такі етапи:
1. Завантажено дані про пожежні станції США з використанням бібліотеки GeoPandas.
2. Виконано розрахунок відстаней між пожежними станціями за допомогою формули Haversine, яка враховує кривизну Землі.
3. Результат розрахунку: середня відстань між станціями становить {average_distance:.2f} метрів.
4. Візуалізовано розташування всіх станцій та вибраних станцій на карті.

Програма структурована у вигляді функцій, що забезпечує читабельність і можливість повторного використання коду.
'''
