import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from alive_progress import alive_bar


def SVM(train_data, train_labels, test_data, test_labels):

    # classifier = svm.SVC(kernel='linear')

    # classifier.fit(train_data, train_label)

    # pred = classifier.predict(test_data)

    # print(pred)

    # print("Accuracy:", accuracy_score(test_label, pred))

    C = 1  # SVM regularization parameter
    models = np.array([svm.SVC(kernel='rbf', C=C),
                       svm.SVC(kernel='poly', degree=2, C=C),
                       svm.SVC(kernel='poly', degree=3, C=C),
                       svm.SVC(kernel='poly', degree=4, C=C),
                       svm.SVC(kernel='poly', degree=5, C=C),
                       svm.SVC(kernel='poly', degree=6, C=C)])

    predicts = []
    with alive_bar(models.shape[0], force_tty=True, title="Testing", bar="classic", theme="classic") as bar:
        for clf in models:
            pred = clf.fit(train_data, train_labels).predict(test_data)
            predicts.append(accuracy_score(test_labels, pred))
            print(f"Accuracy for: {clf}", accuracy_score(test_labels, pred))
            bar()

    print(
        f"The highest accuracy is {np.max(predicts)} which from {models[np.where(predicts==np.max(predicts))[0][0]]}")
    print("Done")
