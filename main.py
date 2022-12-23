import pandas as pd
# import numpy as np
# from importlib.machinery import SourceFileLoader

import src.landmarks as lm
from A1.Training_Models import Model_Training_Testing


def Load_data(type):
    if type == "A":
        print(
            "==============================Loading data for Task A=============================\n"
        )
        # training_datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23/celeba/labels.csv', sep='\t', index_col=0)
        # testing_datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23_test/celeba_test/labels.csv', sep='\t', index_col=0)
        # with Loader("Loading the dataset..."):
        features_train, train_labels = lm.extract_features_labels("Train")
        features_test, test_labels = lm.extract_features_labels("Test")
        print(
            "\n==============================Data loading complete===============================\n"
        )
        return features_train, train_labels, features_test, test_labels
    else:
        datasets = pd.read_csv(
            "./Datasets/dataset_AMLS_22-23/cartoon_set/labels.csv",
            sep="\t",
            index_col=0,
        )
        return datasets


def solve_A1():
    print(
        "==============================Task A1 start to solve=============================="
    )

    features_train, train_labels, features_test, test_labels = Load_data("A")

    print(
        "=========================+====Training the model selected========================="
    )
    # Supervised feature selection method:
    # Support vector machine (SVM), K-nearest neighbors (KNN), Random forest (RF) and Adaboost
    Model_Options = ['SVM', 'KNN', 'RF', 'AdaBoost']
    for model in Model_Options:
        Model_Training_Testing(features_train.reshape((features_train.shape[0], features_train.shape[1]*features_train.shape[2])), train_labels[0],
                               features_test.reshape((features_test.shape[0], features_test.shape[1]*features_test.shape[2])), test_labels[0], model)

    return None


def solve_A2():

    return None


def solve_B1(datasets):

    return None


def solve_B2(datasets):

    return None


def main():
    # For part A
    solve_A1()
    solve_A2()

    # For part B
    datasets = Load_data("B")
    solve_B1(datasets)
    solve_B2(datasets)


if __name__ == "__main__":
    main()
