import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from alive_progress import alive_bar
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix



def SVM(train_data, train_labels, test_data, test_labels):

    # SVM regularization parameter
    C = list([1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3])

    # Tunning the hyperparameter
    cv_scores = []
    with alive_bar(len(C), force_tty=True, title='Training', bar='classic', theme='classic') as bar:
        for c in C:
            model = svm.LinearSVC(C=c, dual=False)
            scores = cross_val_score(
                model, train_data, train_labels, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())
            bar()

    # plt.plot(range(len(C)), cv_scores)
    # plt.xlabel('C')
    # plt.ylabel('Accuracy')
    # plt.show()

    best_C = C[np.where(cv_scores == max(cv_scores))[0][0]]
    print(f'The best regularization parameter is {best_C}')
    
    # Training the best model
    best_svm = svm.LinearSVC(C=best_C, dual=False)
    best_svm.fit(train_data, train_labels)
    svm_pred = best_svm.predict(test_data)
    
    # Validation
    print('SVC precision : {:.2f}'.format(precision_score(svm_pred, test_labels)))
    print('SVC recall    : {:.2f}'.format(recall_score(svm_pred, test_labels)))
    print('Accuracy Score: {:.2f}'.format(accuracy_score(test_labels, svm_pred)))
    print('SVC f1-score  : {:.2f}'.format(f1_score(svm_pred, test_labels)))
    
    confusion_mat = confusion_matrix(test_labels, svm_pred, labels=[1, 0])
    print('This is the confusion matrix:')
    plt.figure(num='Confusion Matrix A1 SVM')
    plt.title('Confusion Matrix A1 SVM')
    ax = sns.heatmap((confusion_mat / np.sum(confusion_mat)*100),
                annot=True, fmt=".2f", cmap="crest")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()
    