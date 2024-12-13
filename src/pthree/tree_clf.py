import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import git
import warnings
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from pthree.create_dataset import feature_selection_ecotaxa, feature_selection_dino_pca, feature_selection_dino_cpics
from pthree.plot import set_plt_params


PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir


def decision_tree_planktoscope(data="metadata", random_state=42):
    if data == "dino":
        file_path = f"{PATH_TO_ROOT}/data/dino/dinov2_features.csv"
        X, y, label_dict = feature_selection_dino_pca(file_path, incl_labels=True)

    else:
        file_path = f"{PATH_TO_ROOT}/data/metadata/ecotaxa_full.csv"
        X, y, label_dict = feature_selection_ecotaxa(file_path, incl_labels=True)

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

    ft_imp = pd.DataFrame(clf.feature_importances_,
             index=X_train.columns,
             columns=["importance"]
    ).sort_values("importance", ascending=False)
    
    save_ft = f"{PATH_TO_ROOT}/models/ft_imp/dt_{data}.csv"
    ft_imp.to_csv(save_ft)

    # y_labels = list(label_dict.values())

    ConfusionMatrixDisplay.from_estimator(
        clf, 
        X_test, 
        y_test, 
        cmap="mako"
    )
    # plt.xticks(range(len(label_dict)), y_labels, rotation = 45, ha="right")
    # plt.yticks(range(len(label_dict)), y_labels)
    plt.title(f"Decision tree, accuracy = {acc:.3f}")
    plt.savefig(
        f"latex/figures/cm_tree_planktoscope_{data}_labeled.pdf",
        bbox_inches = "tight"
    )

    criterion = clf.get_params()["criterion"]
    return criterion


def adaboost_planktoscope(criterion, data="metadata", random_state=42):
    if data == "dino":
        file_path = f"{PATH_TO_ROOT}/data/dino/dinov2_features.csv"
        X, y, label_dict = feature_selection_dino_pca(file_path, incl_labels=True)

    else:
        file_path = f"{PATH_TO_ROOT}/data/metadata/ecotaxa_full.csv"
        X, y, label_dict = feature_selection_ecotaxa(file_path, incl_labels=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state
    )
    params = {
        "estimator": [
            DecisionTreeClassifier(
                criterion=criterion,
                max_depth=1
            ),
            DecisionTreeClassifier(
                criterion=criterion,
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

    ft_imp = pd.DataFrame(clf.feature_importances_,
             index=X_train.columns,
             columns=["importance"]
    ).sort_values("importance", ascending=False)
    
    save_ft = f"{PATH_TO_ROOT}/models/ft_imp/adaboost_{data}.csv"
    ft_imp.to_csv(save_ft)

    # y_labels = list(label_dict.values())

    ConfusionMatrixDisplay.from_estimator(
        clf, 
        X_test, 
        y_test, 
        cmap="mako"
    )
    plt.title(f"Adaboost, accuracy = {acc:.3f}")
    # plt.xticks(range(len(label_dict)), y_labels, rotation = 45, ha="right")
    # plt.yticks(range(len(label_dict)), y_labels)
    plt.savefig(
        f"latex/figures/cm_adaboost_planktoscope_{data}_labeled.pdf", 
        bbox_inches = "tight"
    )
    fig, ax = plt.subplots()
    ax.plot(
        boosting_errors["Number of trees"], 
        boosting_errors["AdaBoost"],
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    plt.savefig(
        f"latex/figures/be_adaboost_planktoscope_{data}.pdf",
        bbox_inches = "tight"
    )


def decision_tree_cpics(random_state=42):
    file_path = f"{PATH_TO_ROOT}/data/dino/222k_all_dino2_features.csv"
    X, y = feature_selection_dino_cpics(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=10,
        min_samples_split=5
    )
    clf.fit(X_train, y_train)
    
    print(f"Dataset: cpics dinov2 features")
    acc_train = clf.score(X_train, y_train)
    print("Results on training data:")
    print(f"\tAccuracy = {acc_train}")

    acc = clf.score(X_test, y_test)
    print("Result on test data:")
    print(f"\tAccuracy = {acc}")

    ft_imp = pd.DataFrame(clf.feature_importances_,
             index=X_train.columns,
             columns=["importance"]
    ).sort_values("importance", ascending=False)
    
    save_ft = f"{PATH_TO_ROOT}/models/ft_imp/dt_cpics.csv"
    ft_imp.to_csv(save_ft)


def adaboost_cpics(random_state=42):
    file_path = f"{PATH_TO_ROOT}/data/dino/222k_all_dino2_features.csv"
    X, y = feature_selection_dino_cpics(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state
    )
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(
            criterion="entropy",
            max_depth=1
        ),
        n_estimators=200, 
        learning_rate=1
    )
    clf.fit(X_train, y_train)
    n_est = clf.get_params()["n_estimators"]

    boosting_errors = {
        "Number of trees": range(1, n_est + 1),
        "AdaBoost": [
            1 - accuracy_score(y_val, y_pred)
            for y_pred in clf.staged_predict(X_val)
        ],
    }
    acc_train = clf.score(X_train, y_train)
    print("Results on training data:")
    print(f"\tAccuracy = {acc_train}")
    
    acc = clf.score(X_test, y_test)
    print("Result on test data:")
    print(f"\tAccuracy = {acc}")

    fig, ax = plt.subplots()
    ax.plot(
        boosting_errors["Number of trees"], 
        boosting_errors["AdaBoost"],
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    plt.savefig(f"latex/figures/be_adaboost_cpics_dino.pdf")


if __name__ == '__main__':
    # Ignores SKlearn warning on split size
    warnings.filterwarnings('ignore') 
    random_state = 2024
    set_plt_params()

    print("Decision tree")
    criterion = decision_tree_planktoscope(
        random_state=random_state
    )
    print(f"Criterion = {criterion}")

    print("AdaBoost")
    adaboost_planktoscope(
        criterion=criterion, 
        random_state=random_state
    )

    print("Decision tree")
    criterion = decision_tree_planktoscope(
        data="dino",
        random_state=random_state
    )
    print("AdaBoost")
    adaboost_planktoscope(
        criterion=criterion,
        data="dino",
        random_state=random_state
    )
    # print("Decision tree")
    # criterion = decision_tree_cpics(
    #     random_state=random_state
    # )
    # print("Adaboost")
    # adaboost_cpics(
    #     random_state=random_state
    # )


"""
Dataset: metadata
Method: decision tree
Results on training data:
        Parameters = {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 20}
        Accuracy = 0.7828358208955224
Result on test data:
        Accuracy = 0.7797619047619048

Method: adaboost
Results on training data:
        Parameters = {'estimator': DecisionTreeClassifier(max_depth=2), 'learning_rate': 1, 'n_estimators': 1000}
        Accuracy = 0.8567164179104477
Result on test data:
        Accuracy = 0.8422619047619048

Dataset: dino
Method: decision tree
Results on training data:
        Parameters = {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 5}
        Accuracy = 0.7172294372294372
Result on test data:
        Accuracy = 0.7288135593220338

Method: adaboost
Results on training data:
        Parameters = {'estimator': DecisionTreeClassifier(criterion='entropy', max_depth=2), 'learning_rate': 1, 'n_estimators': 1000}
        Accuracy = 0.9061740890688259
Result on test data:
        Accuracy = 0.8910411622276029

Dataset: cpics dinov2 features
Method: decision tree
Results on training data:
        Accuracy = 0.6069426643400917
Result on test data:
        Accuracy = 0.5642647024760846
        """