import pandas as pd
import numpy as np
from importlib.machinery import SourceFileLoader

import src.landmarks as lm


def Load_data(type):
    if type == "A":
        print(
            "==============================Loading data for Task A=============================="
        )
        features, labels = [], []
        features_train, labels_train = lm.extract_features_labels("Train")
        features_test, labels_test = lm.extract_features_labels("Test")
        features.extend([features_train, features_test])
        labels.extend([labels_train, labels_test])
        print(
            "==============================Data loading complete=============================="
        )
        return features, labels
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
    features, labels = Load_data("A")
    a = 1
    # Supervised feature selection method:
    # Random forests
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
