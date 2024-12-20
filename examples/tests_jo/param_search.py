import numpy as np
import matplotlib.pyplot as plt
import git
import warnings
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from pthree.create_dataset import feature_selection_ecotaxa, feature_selection_dino_pca


PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir


def decision_tree_planktoscope(data="metadata", random_state=2024):
    if data == "dino":
        path_file = f"{PATH_TO_ROOT}/data/dino/dinov2_features.csv"
        X, y = feature_selection_dino_pca(path_file)

    else:
        path_file = f"{PATH_TO_ROOT}/data/metadata/ecotaxa_full.csv"
        X, y = feature_selection_ecotaxa(path_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    params = {
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [5, 10, 15, 20],
        "criterion": ["gini", "entropy"]
    }
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_grid=params,
        cv=5, # Decrease from 10 to 5 to lower computational cost
        scoring="accuracy",
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Dataset: {data}")
    print("Results on training data:")
    print(f"\tParameters = {grid_search.best_params_}")
    print(f"\tAccuracy = {grid_search.best_score_}")

    clf = grid_search.best_estimator_
    acc = clf.score(X_test, y_test)
    print("Result on test data:")
    print(f"\tAccuracy = {acc}")

    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.title(f"Decision tree, accuracy = {acc:.3f}")
    plt.savefig("latex/figures/cm_tree_planktoscope.pdf")


def adaboost_planktoscope(data="metadata", random_state=2024):
    if data == "dino":
        path_file = f"{PATH_TO_ROOT}/data/dino/dinov2_features.csv"
        X, y = feature_selection_dino_pca(path_file)

    else:
        path_file = f"{PATH_TO_ROOT}/data/metadata/ecotaxa_full.csv"
        X, y = feature_selection_ecotaxa(path_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state
    )
    params = {
        "estimator": [
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1
            ),
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=2
            )
        ],
        "n_estimators": [100, 500, 1000], # [100]
        "learning_rate": [0.001, 0.01, 0.1, 1] # [1]
    }
    grid_search = GridSearchCV(
        estimator=AdaBoostClassifier(
            algorithm="SAMME", 
            random_state=random_state
        ),
        param_grid=params,
        cv=5, # Decrease from 10 to 5 to lower computational cost
        scoring="accuracy", 
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print(f"Dataset: {data}")
    print("Results on training data:")
    print(f"\tParameters = {grid_search.best_params_}")
    print(f"\tAccuracy = {grid_search.best_score_}")

    clf = grid_search.best_estimator_
    n_est = clf.get_params()["n_estimators"]

    boosting_errors = {
        "Number of trees": range(1, n_est + 1),
        "AdaBoost": [
            1 - accuracy_score(y_val, y_pred)
            for y_pred in clf.staged_predict(X_val)
        ],
    }
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Result on test data:")
    print(f"\tAccuracy = {acc}")

    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.title(f"Adaboost, accuracy = {acc:.3f}")
    plt.savefig("latex/figures/cm_adaboost_planktoscope.pdf")

    fig, ax = plt.subplots()
    ax.plot(
        boosting_errors["Number of trees"], 
        boosting_errors["AdaBoost"],
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    plt.savefig("latex/figures/be_adaboost_planktoscope.pdf")


def param_search_tree(data="metadata", random_state=2024):
    PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
    directory = f"{PATH_TO_ROOT}/data/"
    
    if data == "dino":
        path_file = f"{directory}dino/dinov2_features.csv"
        X, y = feature_selection_dino_pca(path_file)

    else:
        path_file = f"{directory}metadata/ecotaxa_full.csv"
        X, y = feature_selection_ecotaxa(path_file)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    param_grid = {
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [5, 10, 15, 20],
        "criterion": ["gini", "entropy"]
    }
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5, # Decrease from 10 to 5 to lower computational cost
        scoring="accuracy",
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print(f"Dataset: {data}")
    print("Results on training data:")
    print(f"\tParameters = {grid_search.best_params_}")
    print(f"\tAccuracy = {grid_search.best_score_}")

    clf = grid_search.best_estimator_
    acc = clf.score(X_test, y_test)
    
    print("Result on test data:")
    print(f"\tAccuracy = {acc}")

    print("Save best model...")
    if data == "dino":
        filename = "models/dino_tree.pkl"
    else:
        filename = "models/metadata_tree.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    print("Done!")


def param_search_adaboost_entropy(data="metadata", random_state=2024):
    PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
    directory = f"{PATH_TO_ROOT}/data/"
    
    if data == "dino":
        path_file = f"{directory}dino/dinov2_features.csv"
        X, y = feature_selection_dino_pca(path_file)

    else:
        path_file = f"{directory}metadata/ecotaxa_full.csv"
        X, y = feature_selection_ecotaxa(path_file)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    weak1 = DecisionTreeClassifier(criterion="entropy", max_depth=1)
    weak2 = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    # Drop depth 3 due to computational complexity
    # weak3 = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    param_grid = {
        "estimator": [weak1, weak2],
        "n_estimators": [100, 500, 1000],
        "learning_rate": [0.001, 0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(
        estimator=AdaBoostClassifier(algorithm="SAMME", random_state=random_state),
        param_grid=param_grid,
        cv=5, # Decrease from 10 to 5 to lower computational cost
        scoring="accuracy", 
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print(f"Dataset: {data}")
    print("Results on training data:")
    print(f"\tParameters = {grid_search.best_params_}")
    print(f"\tAccuracy = {grid_search.best_score_}")

    clf = grid_search.best_estimator_
    acc = clf.score(X_test, y_test)
    print("Result on test data:")
    print(f"\tAccuracy = {acc}")

    print("Save best model...")
    if data == "dino":
        filename = "models/dino_adaboost_entropy.pkl"
    else:
        filename = "models/metadata_adaboost_entropy.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    print("Done!")


def param_search_adaboost_gini(data="metadata", random_state=2024):
    PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
    directory = f"{PATH_TO_ROOT}/data/"
    
    if data == "dino":
        path_file = f"{directory}dino/dinov2_features.csv"
        X, y = feature_selection_dino_pca(path_file)

    else:
        path_file = f"{directory}metadata/ecotaxa_full.csv"
        X, y = feature_selection_ecotaxa(path_file)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    weak1 = DecisionTreeClassifier(criterion="gini", max_depth=1)
    weak2 = DecisionTreeClassifier(criterion="gini", max_depth=2)
    # weak3 = DecisionTreeClassifier(criterion="gini", max_depth=3)

    param_grid = {
        "estimator": [weak1, weak2],
        "n_estimators": [100, 500, 1000],
        "learning_rate": [0.001, 0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(
        estimator=AdaBoostClassifier(algorithm="SAMME", random_state=random_state),
        param_grid=param_grid,
        cv=5, # Decrease from 10 to 5 to lower computational cost
        scoring="accuracy", 
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print(f"Dataset: {data}")
    print("Results on training data:")
    print(f"\tParameters = {grid_search.best_params_}")
    print(f"\tAccuracy = {grid_search.best_score_}")

    clf = grid_search.best_estimator_
    acc = clf.score(X_test, y_test)
    print("Result on test data:")
    print(f"\tAccuracy = {acc}")

    print("Save best model...")
    if data == "dino":
        filename = "models/dino_adaboost_gini.pkl"
    else:
        filename = "models/ecotaxa_adaboost_gini.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    print("Done!")


if __name__ == '__main__':
    # Ignores SKlearn warning on split size
    warnings.filterwarnings('ignore') 
    adaboost_planktoscope()

    # print("Decision tree - Metadata")
    # param_search_tree("metadata")
    # print("Decision tree - Dino")
    # param_search_tree("dino")
    # print("Adaboost entropy - Metadata")
    # param_search_adaboost_entropy("metadata")
    # print("Adaboost entropy - Dino")
    # param_search_adaboost_entropy("dino")


""" 
--------------------------------------
5-fold cross-validation Decision tree
--------------------------------------
Metadata
Results on training data:
        Parameters = {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 5}
        Accuracy = 0.7994972490988428
Result on test data:
        Accuracy = 0.8186157517899761

Dino
Results on training data:
        Parameters = {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 5}
        Accuracy = 0.7288025889967639
Result on test data:
        Accuracy = 0.7034883720930233


-------------------------------------------
5-fold cross-validation AdaBoost (entropy)
-------------------------------------------
Metadata
Results on training data:
        Parameters = {'estimator': DecisionTreeClassifier(criterion='entropy', max_depth=2), 'learning_rate': 1, 'n_estimators': 1000}
        Accuracy = 0.7780117624739139
Result on test data:
        Accuracy = 0.8042959427207638

Dino
Results on training data:
        Parameters = {'estimator': DecisionTreeClassifier(criterion='entropy', max_depth=2), 'learning_rate': 1, 'n_estimators': 1000}
        Accuracy = 0.9158576051779935
Result on test data:
        Accuracy = 0.8972868217054264
"""