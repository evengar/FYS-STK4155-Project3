import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def tree_baseline(random_state=2024):
    cancer = load_breast_cancer()
    clf = DecisionTreeClassifier(random_state=random_state, criterion='gini')

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc}")


def boosting_baseline(random_state=2024):
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
    print(f"Model accuracy: {acc}")

# Set up based on SKlearn API reference
def get_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


def performance_adaboost(random_state=2024):
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
    boosting_errors = {
        "Number of trees": range(1, n_est + 1),
        "AdaBoost": [
            get_error(y_test, y_pred)
            for y_pred in clf.staged_predict(X_test)
        ],
    }

    fig, ax = plt.subplots()
    ax.plot(
        boosting_errors["Number of trees"], 
        boosting_errors["AdaBoost"],
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    plt.show()


if __name__ == '__main__':
    # tree_baseline()
    # boosting_baseline()
    performance_adaboost()