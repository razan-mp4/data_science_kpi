# --------------------------- Lab_work_8 ------------------------------------

'''
Виконав: Одемчук Назар
Lab_work_8, ІІ рівень складності:
Розробити програмний скрипт, що реалізує:
1. Скоринговий аналіз позичальників за даними Data_description.xlsx,
Sample_data.xlsx відповідно до моделі баєсовського класифікатора.
2. Передбачити чи буде кредит повернено у форматі бінарної оцінки (0 або 1);
3. Виявлення шахрайства та фальсифікації даних.

'''

# ------------------------------ Imports --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from pyod.models.knn import KNN


# --------------------------------- Gaussian Naive Bayes analysis ----------------------------
def naive_bayes(data, give):
    '''
    :param data: input data(x) for model training and testing
    :param give: output data(y) for model training and testing
    :return:
    '''

    print(data.T)
    print(data.describe())

    # Correlation
    corr = data.corr()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(data, give, test_size=0.2, random_state=0)

    # Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print(f"GaussianNB Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{classification_rep}")
    print(f"Predictions Array:\n{predictions}")

    # Hist
    plt.figure(figsize=(8, 6))
    plt.hist(predictions, bins=[0, 0.5, 1], edgecolor='black')
    plt.xlabel('Прогноз')
    plt.ylabel('Частота')
    plt.title('Передбачення повернення кредиту')
    plt.xticks([0, 1], ['Не повернуть', 'Повернуть'])
    plt.show()

    # Visualising the Actual and predicted Result
    plt.plot(predictions, color='blue', label='Predicted')
    plt.plot(np.array(y_test), color='red', label='Actual')
    plt.grid(alpha=0.3)
    plt.xlabel('Number of Candidate')
    plt.ylabel('Score')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()

    return


# -------------------------------- Data preparation (підготовка) ---------------------------------

def data_preparation(filname):
    '''
    :param filename: path to file
    :return: proceed dataframe
    '''

    descriptions = pd.read_excel(filname)
    client_bank_descriptions = descriptions[
        (descriptions.Place_of_definition == 'Вказує позичальник') |
        (descriptions.Place_of_definition == 'параметри, повязані з виданим продуктом')
        ]
    client_bank_fields = client_bank_descriptions["Field_in_data"]

    data = pd.read_excel("sample_data.xlsx")

    data = data.loc[:, list(set(client_bank_fields).intersection(data.columns))]
    data = data.dropna(axis=1)
    data.head()

    return data


# --------------------------- Data labelling (маркування) ------------------------------------------
def clip_data(data):
    '''
    :param data: data for clip-ing
    :return: proceed data
    '''

    return np.where(np.sign(data) >= 0, 1, -1) * np.clip(np.abs(data), 1e-9, None)


def min_max(data):
    '''
    :param data: data for min/max-ing
    :return: proceed data
    '''

    return (data - data.min()) / (data.max() - data.min())


def voronin(data: pd.DataFrame, weights: np.array, direction: np.array) -> np.array:
    '''
    :param data: input data
    :param weights: calculated weight for each criteria
    :param direction: direction for each criteria(depends on min/max value)
    :return: calculated score
    '''

    data = data.copy()
    data.loc[direction] = 1 / clip_data(data[direction].values)
    data = data.values
    criteria_sum = np.sum(data, axis=1, keepdims=True)
    normalized_criteria_values = data / clip_data(criteria_sum)
    integro = np.dot(weights, 1 / (1 - normalized_criteria_values))
    return integro


# ------------------------- Fraud detection (шахрайство) -----------------------------------
def Fraud_detection(minimax_data):
    '''
    :param data: input data
    :return:
    '''

    outliers_fraction = 0.1
    X = minimax_data.T
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)

    # -------------------- Fraud detection (шахрайство) PLOT -------------------------------
    norm_data = StandardScaler().fit_transform(minimax_data.T)
    compressed = PCA(n_components=2).fit_transform(norm_data)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=compressed[:, 0], y=compressed[:, 1], hue=np.where(y_pred, "fraud", "no fraud"))
    plt.title('Fraud_detection')
    plt.show()
    return


# -------------------- fraud, no fraud (повернення) PLOT -------------------------------
def fraud_no_fraud(minimax_data):
    '''
    :param data: input data
    :return:
    '''

    outliers_fraction = 0.1
    X = minimax_data.T
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)

    # ------------------------------ Binary scores -------------------------------------
    credit_given = minimax_data.T[data["give"] & ~y_pred].reset_index(drop=True)

    outliers_fraction = 0.026
    X = credit_given
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)

    # -----------------------------------------------------------------------------------

    norm_data = StandardScaler().fit_transform(credit_given)
    compressed = PCA(n_components=2).fit_transform(norm_data)
    plt.figure(figsize=(10, 5))

    # -------------------- fraud, no fraud (повернення) PLOT -------------------------------
    sns.scatterplot(x=compressed[:, 0], y=compressed[:, 1],
                    hue=np.where(y_pred, "will not return", "will return"))
    plt.title('fraud_no_fraud')
    plt.show()
    return


# ----------------------------------------- main ---------------------------------------

if __name__ == '__main__':
    data = data_preparation("data_description.xlsx")
    print(data)
    minimax_info = pd.read_excel("d_segment_data_description_cleaning_minimax.xlsx")
    minimax_info = minimax_info[["Field_in_data", "Minimax"]]
    minimax_info = minimax_info.dropna()
    minimax_info.head()
    print(minimax_info.head())

    # ------------------------------------ minimax_data ---------------------------------
    minimax_data = data.T
    col_intersection = list(set(data.columns).intersection(minimax_info["Field_in_data"]))
    minimax_data = minimax_data.loc[col_intersection, :]
    minimax_data.head()

    criteria_count = len(minimax_data)
    criteria_values = minimax_data.astype(float)
    criteria_values = criteria_values.reset_index(drop=True)
    direction = (minimax_info.set_index("Field_in_data").loc[minimax_data.index, :]["Minimax"] == "max").values

    # -------------------------------------- integro - SCOR -----------------------------
    integro = min_max(voronin(criteria_values, np.ones(criteria_count) / criteria_count, direction))

    # ------------------------------------------- plot -----------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(np.sort(integro.clip(0, 0.2)), label="credit scores")
    plt.hlines(0.0185, 0, len(integro), color="r",
                label=f"threshold ({(integro <= 0.0185).sum() / len(integro):.3%})", )
    plt.legend()
    plt.grid()
    plt.show()

    # ------------------------------------------- scor_d_line -----------------------------------
    scor_d_line = data["give"] = integro <= 0.0185  # емпірічна константа відсікання рішення
    # np.savetxt('Integro_Scor.txt', scor_d_line)       # файл інтегрованого показника - СКОРУ
    print('scor_d_line= ', scor_d_line)
    print(data)

    # ------------------------------ Naive bayes analysis -------------------------------
    naive_bayes(minimax_data.T, data['give'])

    # ------------------------------- Fraud detection (шахрайство) ------------------------------
    Fraud_detection(minimax_data)

    # -------------------------------- fraud, no fraud (повернення) -----------------------------
    fraud_no_fraud(minimax_data)
