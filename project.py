import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_files

def load_train_data():

    train_data = pd.read_csv('/content/train.csv', header = 0)

    X_train = train_data.drop(columns='fake')
    y_train = train_data['fake']

    return X_train, y_train
    from sklearn.datasets import load_files

def load_test_data():

    test_data = pd.read_csv('/content/test.csv', header = 0)

    X_test = test_data.drop(columns='fake')
    y_test = test_data['fake']

    return X_test, y_test

    def print_grid_search_result(grid_search):

    print(grid_search.best_params_)

    best_train = grid_search.cv_results_["mean_train_score"][grid_search.best_index_]
    print("best mean_train_score: {:.3f}".format(best_train))

    best_test = grid_search.cv_results_["mean_test_score"][grid_search.best_index_]
    print("best mean_test_score: {:.3f}".format(best_test))
    from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_actual, y_pred, labels, title='confusion matrix'):

    data = confusion_matrix(y_actual, y_pred)
    ax = sns.heatmap(data,
                     annot=True,
                     cbar=False,
                     fmt='d',
                     xticklabels = labels,
                     yticklabels = labels)
    ax.set_title(title)
    ax.set_xlabel("predicted values")
    ax.set_ylabel("actual values")
    plt.show()
    X_data, y_data = load_train_data()
print(X_data.info())

X_data, y_data = load_train_data()
print(X_data.info())

X_data.head()

print("Size: ",X_data.shape, ", Type: ", type(X_data))
print("Size: ",y_data.shape, ", Type: ", type(y_data))

data_corr = X_data.corr(method='pearson')
ax = sns.heatmap(data_corr, vmin=-1, vmax=1, cmap='BrBG')
ax.set_title("Correlation Heatmap Between Features")

print(X_data.isnull().sum())
unique, freq = np.unique(y_data, return_counts = True)

for i, j in zip(unique, freq):
    print("Label: ", i, ", Frequency: ", j)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=37)

print(X_train.shape)
print(y_train.shape)

model = GradientBoostingClassifier(max_depth=5, random_state=56)

parameters = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0]}


grid2 = GridSearchCV(model, parameters, cv=7, scoring='average_precision', return_train_score=True)
grid1.fit(X_train, y_train)

print_grid_search_result(grid1)

from sklearn.model_selection import GridSearchCV
import os

model = RandomForestClassifier(random_state=55)

parameters = {'n_estimators': [300, 500, 700, 1000],
              'max_depth': [7, 9, 11, 13]}

grid1 = GridSearchCV(model, parameters, cv=7, scoring='average_precision',return_train_score=True)

grid2.fit(X_train, y_train)
print_grid_search_result(grid2)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([('preprocessing', StandardScaler()), ('classifier', grid1.best_estimator_)])
pipeline.fit(X_train, y_train)

labels = ["genuine", "fake"]
title = "Predicting Fake Instagram Account"
plot_confusion_matrix(y_final, y_pred, labels, title)