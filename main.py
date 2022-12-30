# import pandas as pd
# import numpy as np
# from importlib.machinery import SourceFileLoader

import A1.landmarks as lm
import A2.feature_extraciton as fx
from A1.Training_Models_A1 import Model_Training_Testing_A1
from A2.Training_Models_A2 import Model_Training_Testing_A2


def Load_data(type):
    if type == "A":
        print(
            "==============================Loading data for Task A=============================\n"
        )

        features_train, train_labels = lm.extract_features_labels("Train")
        features_test, test_labels = lm.extract_features_labels("Test")

        print(
            "\n==============================Data loading complete===============================\n"
        )
        return features_train, train_labels, features_test, test_labels
    else:
        print(
            "==============================Loading data for Task B=============================\n"
        )

        features_train, train_labels = fx.extract_features_labels("Train")
        features_test, test_labels = fx.extract_features_labels("Test")

        print(
            "\n==============================Data loading complete===============================\n"
        )
        return features_train, train_labels, features_test, test_labels


def solve_A1(features_train, train_labels, features_test, test_labels):
    # Supervised feature selection method:
    # Support vector machine (SVM), K-nearest neighbors (KNN), Random forest (RF) and Adaboost
    Model_Options = ['SVM', 'KNN', 'RF', 'AdaBoost']
    for model in Model_Options:
        Model_Training_Testing_A1(features_train['A1'].reshape((features_train['A1'].shape[0], features_train['A1'].shape[1]*features_train['A1'].shape[2])), train_labels[0],
                                  features_test['A1'].reshape((features_test['A1'].shape[0], features_test['A1'].shape[1]*features_test['A1'].shape[2])), test_labels[0], model)
    print("==============================Task A1 Completed!==============================")
    return None


def solve_A2(features_train, train_labels, features_test, test_labels):
    # Supervised feature selection method:
    # Support vector machine (SVM), K-nearest neighbors (KNN), Random forest (RF) and Adaboost
    Model_Options = ['SVM', 'KNN', 'RF', 'AdaBoost']
    for model in Model_Options:
        Model_Training_Testing_A2(features_train['A2'].reshape((features_train['A2'].shape[0], features_train['A2'].shape[1]*features_train['A2'].shape[2])), train_labels[1],
                                  features_test['A2'].reshape((features_test['A2'].shape[0], features_test['A2'].shape[1]*features_test['A2'].shape[2])), test_labels[1], model)
    print("==============================Task A2 Completed!==============================")
    return None


def solve_B1(features_train, train_labels, features_test, test_labels):

    return None


def solve_B2(features_train, train_labels, features_test, test_labels):

    return None


def main():
    # For part A
    # Loading Data
    print("==============================Task A1 start to solve==============================")
    features_train, train_labels, features_test, test_labels = Load_data("A")

    # A1
    print("==============================Training the model selected=========================")
    solve_A1(features_train, train_labels, features_test, test_labels)

    # A2
    print("==============================Task A2 start to solve==============================")
    print("==============================Training the model selected=========================")
    solve_A2(features_train, train_labels, features_test, test_labels)

    # For part B
    # Loading Data
    print("==============================Task B1 start to solve==============================")
    datasets = Load_data("B")

    # B1
    print("==============================Training the model selected=========================")
    solve_B1(features_train, train_labels, features_test, test_labels)

    # B2
    print("==============================Task B2 start to solve==============================")
    print("==============================Training the model selected=========================")
    solve_B2(features_train, train_labels, features_test, test_labels)


if __name__ == "__main__":
    main()
