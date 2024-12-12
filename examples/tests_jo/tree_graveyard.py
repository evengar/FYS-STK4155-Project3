import numpy as np
import matplotlib.pyplot as plt
import git

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

from pthree.create_dataset import feature_selection_ecotaxa, feature_selection_dino, feature_selection_dino_pca


def tree_cancer(random_state=2024):
    cancer = load_breast_cancer()
    clf = DecisionTreeClassifier(random_state=random_state, criterion='gini')

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Model accuracy: {acc}")


def boosting_cancer(random_state=2024):
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

    weak_learner = DecisionTreeClassifier(max_depth=1)
    n_est = 500

    clf = AdaBoostClassifier(
        estimator=weak_learner, 
        n_estimators=n_est, 
        algorithm='SAMME', 
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Model performance")
    print(f"\tAccuracy = {acc}")
    print(f"\tRecall = {recall}")
    print(f"\tPrecision = {precision}")

# Set up based on SKlearn API reference
def get_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

def tree_ecotaxa(X, y, random_state=2024):
    clf = DecisionTreeClassifier(random_state=random_state, criterion='gini')

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    print(f"Model performance")
    print(f"\tAccuracy = {acc}")
    # print(f"\tRecall = {recall}")
    # print(f"\tPrecision = {precision}")


def tree_dino(X, y, random_state=2024):
    clf = DecisionTreeClassifier(random_state=random_state, criterion='gini')

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)

    tree.plot_tree(clf)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    print(f"Model performance")
    print(f"\tAccuracy = {acc}")
    # print(f"\tRecall = {recall}")
    # print(f"\tPrecision = {precision}")


def performance_adaboost(X, y, random_state=2024):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    weak_learner = DecisionTreeClassifier(max_depth=2)
    n_est = 100

    clf = AdaBoostClassifier(
        estimator=weak_learner, 
        n_estimators=n_est,
        algorithm='SAMME', 
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    boosting_errors = {
        "Number of trees": range(1, n_est + 1),
        "AdaBoost": [
            get_error(y_test, y_pred)
            for y_pred in clf.staged_predict(X_test)
        ],
    }

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    print(f"Model performance")
    print(f"\tAccuracy = {acc}")
    # print(f"\tRecall = {recall}")
    # print(f"\tPrecision = {precision}")

    fig, ax = plt.subplots()
    ax.plot(
        boosting_errors["Number of trees"], 
        boosting_errors["AdaBoost"],
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    plt.show()


def performance_gboost(X, y, random_state=2024):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #weak_learner = DecisionTreeClassifier(max_depth=1)
    n_est = 100

    clf = GradientBoostingClassifier(
        n_estimators=n_est, 
        learning_rate=1.0,
        max_depth=1, 
        random_state=random_state
        )
    clf.fit(X_train, y_train)
    boosting_errors = {
        "Number of trees": range(1, n_est + 1),
        "Gradient Boosting": [
            get_error(y_test, y_pred)
            for y_pred in clf.staged_predict(X_test)
        ],
    }

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    print(f"Model performance")
    print(f"\tAccuracy = {acc}")
    # print(f"\tRecall = {recall}")
    # print(f"\tPrecision = {precision}")

    fig, ax = plt.subplots()
    ax.plot(
        boosting_errors["Number of trees"], 
        boosting_errors["Gradient Boosting"],
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    plt.show()


def performance_randomforest(X, y, random_state=2024):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #weak_learner = DecisionTreeClassifier(max_depth=1)
    n_est = 500

    clf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=2, 
        random_state=random_state)
    clf.fit(X_train, y_train)

    # boosting_errors = {
    #     "Number of trees": range(1, n_est + 1),
    #     "Random Forest": [
    #         clf.score(X_test, y_test)
    #     ],
    # }

    y_pred = clf.predict(X_test)

    acc = clf.score(X_test, y_test)
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    print(f"Model performance")
    print(f"\tAccuracy = {acc}")
    # print(f"\tRecall = {recall}")
    # print(f"\tPrecision = {precision}")

    # fig, ax = plt.subplots()
    # ax.plot(
    #     boosting_errors["Number of trees"], 
    #     boosting_errors["Random Forest"],
    # )
    # ax.set_xlabel("Number of trees")
    # ax.set_ylabel("Error")
    # plt.show()


if __name__ == '__main__':
    PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
    directory = f"{PATH_TO_ROOT}/data/"
    path_file = f"{directory}metadata/ecotaxa_full.csv"
    # tree_baseline()
    # boosting_baseline()
    # performance_adaboost()

    # X, y = feature_selection_ecotaxa(path_file)
    # tree_ecotaxa(X, y)
    # performance_adaboost(X, y)

    # path_dino = f"{directory}dino/dinov2_features.csv"
    # X, y = feature_selection_dino(path_dino)
    # tree_dino(X, y)
    # performance_adaboost(X, y)

    # path_dino = f"{directory}dino/222k_pca_dino2_features.csv"
    # X, y = feature_selection_dino(path_dino)
    # tree_dino(X, y)
    # performance_adaboost(X, y)


    # path_dino = f"{directory}dino/222k_pca_dino2_features.csv"
    # X, y = feature_selection_dino_pca(path_dino)
    # tree_dino(X, y)
    # performance_adaboost(X, y)

    # path_dino = f"{directory}dino/222k_pca_dino2_features.csv"
    # X, y = feature_selection_dino_pca(path_dino)
    # tree_dino(X, y)
    # performance_gboost(X, y)

    path_dino = f"{directory}dino/222k_pca_dino2_features.csv"
    X, y = feature_selection_dino_pca(path_dino)
    tree_dino(X, y)
    performance_randomforest(X, y)
