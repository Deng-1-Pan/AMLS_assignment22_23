import pandas as pd
import numpy as np
from importlib.machinery import SourceFileLoader

import src.landmarks as lm
from A1.SVM import SVM
from src.loading import *


def Load_data(type):
    if type == "A":
        print("==============================Loading data for Task A==============================")
        # training_datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23/celeba/labels.csv', sep='\t', index_col=0)
        # testing_datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23_test/celeba_test/labels.csv', sep='\t', index_col=0)
        # with Loader("Loading the dataset..."):
        features_train, labels_train = lm.extract_features_labels("Train")
        features_test, labels_test = lm.extract_features_labels("Test")
        print("\n")
        print("==============================Data loading complete==============================")
        return features_train, labels_train, features_test, labels_test
    else:
        datasets = pd.read_csv(
            "./Datasets/dataset_AMLS_22-23/cartoon_set/labels.csv",
            sep="\t",
            index_col=0,
        )
        return datasets


def solve_A1():
    print("==============================Task A1 start to solve==============================")

    features_train, labels_train, features_test, labels_test = Load_data("A")

    # Supervised feature selection method:
    # SVM
    A1_SVM_pred = SVM(features_train.reshape((features_train.shape[0], features_train.shape[1]*features_train.shape[2])),
                      labels_train,
                      features_test.reshape(
                          (features_test.shape[0], features_test.shape[1]*features_test.shape[2])),
                      labels_test)
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
