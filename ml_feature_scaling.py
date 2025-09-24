import pandas as pd

dataset = pd.read_csv('Regressão Linear/Desafio/insurance.csv')

print(dataset.shape)
print(dataset.info())

import matplotlib.pyplot as plt

dataset.hist(bins=50, figsize=(35,20))
plt.show()

columns_names = dataset.columns.to_list()  

from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder = LabelEncoder()

for name in columns_names:
    if name not in('sex', 'smoker', 'region'):
        plt.boxplot(dataset[name])
        plt.title(name)
        plt.ylabel('Valores')
        plt.show()
    else:
        dataset[name] = label_encoder.fit_transform(dataset[name])

print(dataset.head())

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np

X = dataset.drop(columns=['charges'])
y = dataset['charges']
dataset['charges_cat'] = pd.cut(
    y, 
    bins=[0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, np.inf],
    labels=[1, 2, 3, 4, 5, 6, 7, 8]
)
y.hist()
plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset['charges_cat']):
    strat_train_set = dataset.iloc[train_index]
    strat_test_set = dataset.iloc[test_index]

print(y.iloc[strat_test_set.index].value_counts() / len(strat_test_set))
print(y.iloc[strat_train_set.index].value_counts() / len(strat_train_set))
print(y.value_counts() / len(dataset))

for set_ in (strat_train_set, strat_test_set):
    set_.drop('charges_cat', axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = strat_train_set.drop(columns=['charges'])
X_test = strat_test_set.drop(columns=['charges'])
y_train = strat_train_set['charges']
y_test = strat_test_set['charges']
# print(strat_train_set)
scaler.fit(X_train)

x_train_santadard_scaled = scaler.transform(X_train)
x_test_santadard_scaled = scaler.transform(X_test)

def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    relative_errors = errors / np.abs(labels)
    mape = np.mean(relative_errors) * 100
    return mape

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

model = DecisionTreeRegressor(max_depth=4)
model.fit(x_train_santadard_scaled, y_train)

y_predictions_standard = model.predict(x_test_santadard_scaled)
accuracy_standard = r2_score(y_test, y_predictions_standard)

print(f'R²: {accuracy_standard*100:.0f}%')
print(f'O MAPE é: {calculate_mape(y_test, y_predictions_standard):.2f}%')
