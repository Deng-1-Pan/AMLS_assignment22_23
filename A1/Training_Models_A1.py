import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def Model_Training_Testing_A1(train_data, train_labels, test_data, test_labels):

    model = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(
        130,), random_state=3, max_iter=5000)

    # Normalize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    # Create a KFold generator with 5 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=5)

    precision_score_list, recall_score_list, accuracy_score_list, f1_score_list = [], [], [], []
    # Using K-FLod to validate training dataset
    with tqdm(total=kfold.get_n_splits(train_data), desc="Validating") as pbar:
        for train_index, test_index in kfold.split(train_data):
            # Get the training and testing data for this fold
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]

            # Train and evaluate a model on the training and testing data for this fold
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            precision_score_list.append(precision_score(pred, y_test))
            recall_score_list.append(recall_score(pred, y_test))
            accuracy_score_list.append(accuracy_score(y_test, pred))
            f1_score_list.append(f1_score(pred, y_test))
            pbar.update(1)

    avg_precision = np.mean(precision_score_list)
    avg_recall = np.mean(recall_score_list)
    avg_accuracy = np.mean(accuracy_score_list)
    avg_f1 = np.mean(f1_score_list)

    print(f'\n Avg precision for MLP: {avg_precision:.4f}')
    print(f'Avg recall for MLP: {avg_recall:.4f}')
    print(f'Avg accuracy for MLP: {avg_accuracy:.4f}')
    print(f'Avg f1 score for MLP: {avg_f1:.4f}\n')

    # Inference stage
    pred_test = model.predict(test_data)

    precision = precision_score(pred_test, test_labels)
    recall = recall_score(pred_test, test_labels)
    accuracy = accuracy_score(test_labels, pred_test)
    f1 = f1_score(pred_test, test_labels)

    print(f'Precision on test dataset for MLP: {precision:.4f}')
    print(f'Recall on test dataset for MLP: {recall:.4f}')
    print(f'Accuracy on test dataset for MLP: {accuracy:.4f}')
    print(f'F1 score on test dataset for MLP: {f1:.4f}\n')

    confusion_mat = confusion_matrix(
        test_labels, pred_test, labels=model.classes_)
    print('This is the confusion matrix:')
    plt.figure(num='Confusion Matrix A1 MLP')
    plt.title('Confusion Matrix A1 MLP')
    ax = sns.heatmap((confusion_mat / np.sum(confusion_mat)*100),
                     annot=True, fmt=".2f", cmap="crest")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()
