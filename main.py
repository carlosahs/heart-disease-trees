import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# Column names from dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

DATA_DIR = "data"

cleveland_data = pd.read_csv(
    os.path.join(DATA_DIR, "cleveland.data"),
    names=columns
)

# Two columns have a '?' value that needs to be replaced
cleveland_data["ca"] = cleveland_data["ca"].replace("?", np.nan)
cleveland_data["thal"] = cleveland_data["thal"].replace("?", np.nan)

# The 'ca' and 'thal' columns should be numeric columns
cleveland_data["ca"] = pd.to_numeric(cleveland_data["ca"])
cleveland_data["thal"] = pd.to_numeric(cleveland_data["thal"])

# Categorical columns from dataset
categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "num"]

# Because of the replaced '?' value, now we have NaN values
cleveland_data = cleveland_data.dropna()  # NaN values are dropped (less than 2% of the data)

# Convert float values to int values for categorical variables
cleveland_data[categorical] = cleveland_data[categorical].astype("int32")

target = "num"  # Diagnosis of heart disease

# Target column contains 4 (0 to 4) categories, however it is reduced to 2 (binary)
# to predict whether a patient has a heart disease or not
cleveland_data[target] = cleveland_data[target] > 0

X = cleveland_data.drop(target, axis=1)
y = cleveland_data[[target]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Model to predict heart disease diagnosis with decision tree
dtree_model = DecisionTreeClassifier(max_depth=5)
dtree_model.fit(X_train, y_train)

y_pred = dtree_model.predict(X_test)
dt_confusion_matrix = confusion_matrix(y_test, y_pred)

print("Decision tree accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:")
print(dt_confusion_matrix)

# # Visualize decision tree model
# diagram = tree.plot_tree(
#     dtree_model,
#     feature_names=X.columns,
#     class_names=["Heart disease", "No heart disease"],
#     filled=True
# )

# plt.show()

# Model to predict heart disease diagnosis with random forest
rforest_model = RandomForestClassifier(
    n_estimators=100,
    max_leaf_nodes=12,
    n_jobs=-1,
    random_state=42
)
rforest_model.fit(X_train, y_train.values.flatten())

y_pred = rforest_model.predict(X_test)
rf_confusion_matrix = confusion_matrix(y_test, y_pred)

print("Random forest accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:")
print(rf_confusion_matrix)

# Hyperparameter optimization for random forest

# Hyperparameters setup
n_estimators = [int(x) for x in np.linspace(start=10, stop=250, num=10)]  # Numbes of trees in forest
max_features = ["sqrt", "log2"]  # How features are sampled at each split
max_leaf_nodes = [None, 6, 12]  # Max number of leaf nodes
max_depth = [None, 4, 8]  # Depth of the tree
bootstrap = [True, False]  # Decide if samples are bootstrapped when building trees

param_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_leaf_nodes": max_leaf_nodes,
    "max_depth": max_depth,
    "bootstrap": bootstrap
}

rforest_grid = GridSearchCV(
    estimator=rforest_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)

rforest_grid.fit(X_train, y_train.values.flatten())

print(rforest_grid.best_params_)

orf_confusion_matrix = confusion_matrix(y_test, y_pred)

# The optimized random forest is less prone to overfitting due to hyperparameter tuning
# and the cross validation of each model in each hyperparameter combination
print("Optimized random forest accuracy:", rforest_grid.score(X_test, y_test))
print("Confusion matrix:")
print(orf_confusion_matrix)

# # Plot confusion matrices
# ConfusionMatrixDisplay(
#     dt_confusion_matrix,
#     display_labels=["heart disease", "no heart disease"]
# ).plot()
# ConfusionMatrixDisplay(rf_confusion_matrix, display_labels=["heart disease", "no heart disease"])
# ConfusionMatrixDisplay(orf_confusion_matrix, display_labels=["heart disease", "no heart disease"])