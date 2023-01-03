import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def Model_Training_Testing_A2(train_data, train_labels, test_data, test_labels):

    model = RandomForestClassifier(random_state=0)

    # Normalize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)
    # Split Train and Validation
    train_data, Validation_data, train_labels, Validation_labdels = train_test_split(
        train_data, train_labels, test_size=0.3, random_state=0)

    # Training the model
    with tqdm(desc="Training") as pbar:
        model.fit(train_data, train_labels)
        pbar.update(1)

    acc_val = accuracy_score(
        Validation_labdels, model.predict(Validation_data))
    print('For RF the validation Accuracy :', acc_val)

    # Inference stage
    pred_test = model.predict(test_data)

    precision = precision_score(pred_test, test_labels)
    recall = recall_score(pred_test, test_labels)
    accuracy = accuracy_score(test_labels, pred_test)
    f1 = f1_score(pred_test, test_labels)

    print(f'Precision on test dataset for RF: {precision:.4f}')
    print(f'Recall on test dataset for RF: {recall:.4f}')
    print(f'Accuracy on test dataset for RF: {accuracy:.4f}')
    print(f'F1 score on test dataset for RF: {f1:.4f}\n')

    confusion_mat = confusion_matrix(
        test_labels, pred_test, labels=model.classes_)
    print('This is the confusion matrix:')
    plt.figure(num='Confusion Matrix A2 RF')
    plt.title('Confusion Matrix A2 RF')
    ax = sns.heatmap((confusion_mat / np.sum(confusion_mat)*100),
                     annot=True, fmt=".2f", cmap="crest")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()
