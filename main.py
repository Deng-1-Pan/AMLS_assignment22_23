import A1.landmarks as lm
import B1.feature_extraction as fx
from A1.Training_Models_A1 import Model_Training_Testing_A1
from A2.Training_Models_A2 import Model_Training_Testing_A2
from B1.Training_Models_B1 import Model_Training_Testing_B1
from B2.Training_Models_B2 import Model_Training_Testing_B2


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

    Model_Training_Testing_A1(features_train.reshape((features_train.shape[0], features_train.shape[1]*features_train.shape[2])), train_labels[0],
                              features_test.reshape((features_test.shape[0], features_test.shape[1]*features_test.shape[2])), test_labels[0])
    print("==============================Task A1 Completed!==============================")
    return None


def solve_A2(features_train, train_labels, features_test, test_labels):

    Model_Training_Testing_A2(features_train.reshape((features_train.shape[0], features_train.shape[1]*features_train.shape[2])), train_labels[1],
                              features_test.reshape((features_test.shape[0], features_test.shape[1]*features_test.shape[2])), test_labels[1])
    print("==============================Task A2 Completed!==============================")
    return None


def solve_B1(features_train, train_labels, features_test, test_labels):

    Model_Training_Testing_B1(features_train['Face'].reshape((features_train['Face'].shape[0], features_train['Face'].shape[1]*features_train['Face'].shape[2])),
                              train_labels[0],
                              features_test['Face'].reshape(
                                  (features_test['Face'].shape[0], features_test['Face'].shape[1]*features_test['Face'].shape[2])),
                              test_labels[0])
    print("==============================Task B1 Completed!==============================")
    return None


def solve_B2(features_train, train_labels, features_test, test_labels):

    Model_Training_Testing_B2(
        features_train['Eyes'], train_labels[1], features_test['Eyes'], test_labels[1])
    print("==============================Task B2 Completed!==============================")

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
    features_train, train_labels, features_test, test_labels = Load_data("B")

    # B1
    print("==============================Training the model selected=========================")
    solve_B1(features_train, train_labels, features_test, test_labels)

    # B2
    print("==============================Task B2 start to solve==============================")
    print("==============================Training the model selected=========================")
    solve_B2(features_train, train_labels, features_test, test_labels)


if __name__ == "__main__":
    main()
