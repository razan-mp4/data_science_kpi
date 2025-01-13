# --------------------------- Lab_work_7 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_7, І рівень складності:
Відповідно до технічних умов, табл.1 додатку.

Розробити програмний скрипт, що реалізує:
1. Парсінг файлу параметрів: Pr_1.xls;
2. Попередній аналіз даних;
3. Визначення показників ефективності – продаж та прибутку;
4. Визначення математичної моделі даних відповідно до МНК;
5. Здійснити прогнозування динаміки зміни продажів на наступні 6 місяців за регіонами
(таблиця, графік).
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# -------- Функція для парсингу вхідних даних --------
def file_parsing(path="Pr_1.xls", additional_path="Pr_1_table.xls"):
    '''
    :param path: path for main data
    :param additional_path: path for help data(associative table)
    :return: proceed dataframe
    '''

    df1 = pd.read_excel(path)
    df2 = pd.read_excel(additional_path)

    df = pd.merge(df1, df2, on='Код магазину', how='outer')
    df['Дата'] = pd.to_datetime(df['Дата'])

    df.to_csv('file_pr.csv')
    print(df)
    print(df.info())

    return df


# -------- Функція для побудови МНК(лінійної) --------
def MNK(data, title):
    '''
    :param data: set of data for OSL
    :param title: plot title
    :return:
    '''

    X = np.linspace(0, len(data), len(data))
    y = data

    res = sm.OLS(y, X).fit()

    print(res.summary())
    print('Parameters: ', res.params)
    print('Standard errors: ', res.bse)
    print('Predicted values: ', res.predict())

    prstd, iv_l, iv_u = wls_prediction_std(res)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(X, y, label="data")
    ax.plot(X, res.fittedvalues, 'r--.', label="OLS")
    ax.plot(X, iv_u, 'r--')
    ax.plot(X, iv_l, 'r--')
    ax.legend(loc='best')
    plt.title(title)
    plt.show()


# -------- Функція для прогнозування наступних n місяців --------
def predict_by_region(monthly_sales, months_number):
    '''
    :param monthly_sales: data about monthly sales
    :param months_number: number of months for prediction
    :return:
    '''

    forecast_table = pd.DataFrame(columns=['Регіон', 'Місяць', 'Прогноз'])

    for region in monthly_sales['Регіон'].unique():
        region_data = monthly_sales[monthly_sales['Регіон'] == region]

        model = SARIMAX(region_data['Продаж'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

        forecast = results.get_forecast(steps=months_number)
        forecast_mean = forecast.predicted_mean.values

        forecast_table = pd.concat(
            [forecast_table, pd.DataFrame({
                'Регіон': [region] * months_number,
                'Місяць': [region_data['Дата'].max() + timedelta(days=31 * i) for i in range(1, months_number + 1)],
                'Прогноз': forecast_mean
            })], ignore_index=True)

    print('Forecast Table')
    print(forecast_table)

    for region in forecast_table['Регіон'].unique():
        region_forecast = forecast_table[forecast_table['Регіон'] == region]

        plt.figure(figsize=(10, 6))
        plt.plot(
            monthly_sales[monthly_sales['Регіон'] == region]['Дата'],
            monthly_sales[monthly_sales['Регіон'] == region]['Продаж'],
            label='Продажі'
        )

        plt.plot(region_forecast['Місяць'], region_forecast['Прогноз'], label='Прогноз')
        plt.title(f'Прогноз продажів для регіону {region}')
        plt.xlabel('Дата')
        plt.ylabel('Продажі')
        plt.legend()
        plt.show()


# -------- Функція для побудови графіків --------
def plot_data(firstData, secondData, text):
    '''
    :param firstData: first graphic
    :param secondData: second graphic
    :param text: text for logs
    :return:
    '''
    plt.clf()
    plt.plot(firstData)
    plt.plot(secondData)
    plt.xlabel(text)
    plt.show()
    return


data = file_parsing()

data['Прибуток'] = data['Кількість реалізацій'] * (data['Ціна реалізації'] - data['Собівартість одиниці'])
data['Продаж'] = data['Кількість реалізацій'] * (data['Собівартість одиниці'])

plot_data(data['Прибуток'], data['Прибуток'], 'Прибуток')
plot_data(data['Продаж'], data['Продаж'], 'Продаж')
plot_data(data['Продаж'], data['Прибуток'], 'Продажі й Прибутки')

MNK(data['Прибуток'], 'Прибуток MNK')

MNK(data['Продаж'], 'Продаж MNK')

monthly_sales = data.groupby(['Регіон', pd.Grouper(key='Дата', freq='M')])['Продаж'].sum().reset_index()
predict_by_region(monthly_sales, 6)

'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Вхідні дані:
- Зчитані дані з файлу Pr_1.xls, об'єднані з асоціативною таблицею за кодами магазинів.
- Дані містять інформацію про продажі, кількість реалізацій, собівартість та ціну реалізації.

2. Ініціалізація:
- Для кожного регіону проведено групування даних за місяцями.
- Ініціалізовано модель SARIMAX для прогнозування.

3. Визначення показників ефективності:
- Розраховано показники прибутку та продажів на основі кількості реалізацій, собівартості та ціни реалізації.

4. МНК:
- Для показників прибутку та продажів побудовано лінійні моделі методом найменших квадратів (МНК).
- Візуалізовано результати побудови моделей.

5. Прогнозування:
- Здійснено прогноз продажів на наступні 6 місяців для кожного регіону за допомогою моделі SARIMAX.
- Побудовано графіки прогнозу для кожного регіону.

6. Результати:
- Виявлено залежності між показниками продажів та прибутку.
- Отримано прогнози продажів на наступні місяці з врахуванням сезонності.

7. Висновок:
- Розроблений скрипт успішно реалізує завдання прогнозування продажів та аналізу ефективності, що підтверджується побудованими графіками та моделями.
- Скрипт забезпечує автоматизацію аналізу даних для управлінських рішень.
'''
