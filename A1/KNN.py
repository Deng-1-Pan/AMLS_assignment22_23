import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from alive_progress import alive_bar
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

def KNN(train_data, train_labels, test_data, test_labels):
    
    k_index = list(range(1,int(np.rint(np.sqrt(len(train_data))))))
    knn_score = []
    
    with alive_bar(len(k_index), force_tty=True, title='Training', bar='classic', theme='classic') as bar:
        for k in k_index:
            model = KNeighborsClassifier(n_neighbors = k) 
            model.fit(train_data,train_labels)
            pred = model.predict(test_data)
            knn_score.append(accuracy_score(pred,test_labels))
            bar()
            
    best_K = k_index[np.where(knn_score == max(knn_score))[0][0]]
    print(f'The best k parameter is {best_K} with average accuracy {knn_score[k_index.index(best_K)]}')

    # Training the best model
    best_knn = KNeighborsClassifier(n_neighbors=best_K)
    best_knn.fit(train_data, train_labels)
    knn_pred = best_knn.predict(test_data)
    
    # Validation
    print('KNN precision : {:.2f}'.format(precision_score(knn_pred, test_labels)))
    print('KNN recall    : {:.2f}'.format(recall_score(knn_pred, test_labels)))
    print('Accuracy Score: {:.2f}'.format(accuracy_score(test_labels, knn_pred)))
    print('KNN f1-score  : {:.2f}'.format(f1_score(knn_pred, test_labels)))
    
    confusion_mat = confusion_matrix(test_labels, knn_pred, labels=[1, 0])
    print('This is the confusion matrix:')
    plt.figure(num='Confusion Matrix A1 SVM')
    plt.title('Confusion Matrix A1 SVM')
    ax = sns.heatmap((confusion_mat / np.sum(confusion_mat)*100),
                annot=True, fmt=".2f", cmap="crest")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()
    