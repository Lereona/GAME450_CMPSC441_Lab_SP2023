""" Lab 10: Save people
You can save people from heart disease by training a model to predict whether a person has heart disease or not.
The dataset is available at src/lab8/heart.csv
Train a model to predict whether a person has heart disease or not and test its performance.
You can usually improve the model by normalizing the input data. Try that and see if it improves the performance. 
"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

data = pd.read_csv("GAME450_CMPSC441_Lab_SP2023\src\lab10\heart.csv")

# Transform the categorical variables into dummy variables.
print(data.head())
string_col = data.select_dtypes(include="object").columns
df = pd.get_dummies(data, columns=string_col, drop_first=False)
print(df.head())

y = df.HeartDisease.values
x = df.drop(["HeartDisease"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=25
)

""" Train a sklearn model here. """

sklearn_model = tree.DecisionTreeClassifier(random_state=1)

sklearn_model.fit(x_train, y_train)

# Accuracy
print("----------------decision tree model----------------")
print("training set accuracy: {}\n".format(sklearn_model.score(x_train, y_train)))
print("Accuracy of decision tree model: {}\n".format(sklearn_model.score(x_test, y_test)))


""" Improve the model by normalizing the input data. """
normalized_df=(df-df.min())/(df.max()-df.min())
y_norm = normalized_df.HeartDisease.values
x_norm = normalized_df.drop(["HeartDisease"], axis=1)
x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(
    x_norm, y_norm, test_size=0.2, random_state=25
)
sklearn_model = GaussianNB()
sklearn_model.fit(x_train_norm, y_train_norm)
print("----------------naive bayes model with normalization----------------")

print("training set accuracy: {}\n".format(sklearn_model.score(x_train_norm, y_train_norm)))
print("Accuracy of naive bayes model: {}\n".format(sklearn_model.score(x_test_norm, y_test_norm)))

""" even better improved model """

sklearn_model = ensemble.RandomForestClassifier()
sklearn_model.fit(x_train, y_train)
print("----------------Random Forest model----------------")
print("training set accuracy: {}\n".format(sklearn_model.score(x_train, y_train)))
print("Accuracy of Random Forest: {}\n".format(sklearn_model.score(x_test, y_test)))


print("----------------Hyperparameter Tuning with Cross Validation on Random Forest model----------------")

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 100)]
max_features = ["sqrt", "log2", None]
criterion = ["gini", "entropy"]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 10, num = 1)]
min_samples_leaf = [int(x) for x in np.linspace(start = 1, stop = 5, num = 1)]
bootstrap = [True, False]
# Create the random grid
grid = {"n_estimators": n_estimators,
        "criterion": criterion,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap}
# #GridSearchCV - takes too long to run on my computer
# gscv = GridSearchCV(estimator = ensemble.RandomForestClassifier(), param_grid = grid, cv = 5, n_jobs = -1)
# gscv.fit(x_train,y_train)
# print("----------------GridSearchCV----------------")
# print("Accuracy of Improved Random Forest: {}\n".format(gscv.score(x_test, y_test)))
# print(gscv.best_params_)
#RandomSearchCV
rscv = RandomizedSearchCV(estimator = ensemble.RandomForestClassifier(), param_distributions = grid, n_iter = 30, cv = 5, random_state=1, n_jobs = -1)
rscv.fit(x_train,y_train)
print("----------------RandomSearchCV----------------")
print("Accuracy of Improved Random Forest: {}\n".format(rscv.score(x_test, y_test)))



