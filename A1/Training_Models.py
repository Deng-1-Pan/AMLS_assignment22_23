import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

random_seed = 0

def model_para(model_index, model, i):
    if model_index == 0:
        model.C = i
    elif model_index == 1:
        model.n_neighbors = i
    elif model_index == 2:
        model.n_estimators = i
    
    return model
        

def Model_Training_Testing(train_data, train_labels, test_data, test_labels, model_name):
    
    if model_name == 'SVM':
        model = svm.LinearSVC(dual=False, random_state=random_seed)
        model_index = 0
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
        model_index = 1
    elif model_name == 'RF':
        model = RandomForestClassifier(random_state=random_seed)
        model_index = 2
    
    # List to store the parameter to be tunned
    list_para = [[] * 1 for _ in range(3)]
    # SVM regularization parameter C
    list_para[0] = [1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3]
    # KNN k parameter
    list_para[1] = list(range(1,int(np.rint(np.sqrt(len(train_data))))))
    # RF n_estimators parameter
    list_para[2] = list(range(268, 274))
    
    

    # Tunning the hyperparameter
    parameter_scores = []
    with tqdm(total=len(list_para[model_index]), desc="Training on " + model_name) as pbar:
        for i in list_para[model_index]:
            train_model = model_para(model_index, model, i)
            scores = cross_val_score(train_model, train_data, train_labels, cv=10, scoring='accuracy')
            parameter_scores.append(scores.mean())
            pbar.update(1)
            
    best_para_index = np.where(parameter_scores == max(parameter_scores))[0][0]
    best_para = list_para[model_index][best_para_index]
    print(f'The best hyperparameter for {model_name} is {best_para} with average accuracy {parameter_scores[list_para[model_index].index(best_para)]}')

    # Training the best model
    best_model = model_para(model_index, model, best_para)
    t1 = time.perf_counter()
    best_model.fit(train_data, train_labels)
    t2 = time.perf_counter()
    print(f"Training Completed! Time used: {t2-t1}s")
    
    # Validation
    best_model_pred = best_model.predict(test_data)
    print('{} precision : {:.2f}'.format(model_name, precision_score(best_model_pred, test_labels)))
    print('{} recall    : {:.2f}'.format(model_name, recall_score(best_model_pred, test_labels)))
    print('{} Accuracy Score: {:.2f}'.format(model_name, accuracy_score(test_labels, best_model_pred)))
    print('{} f1-score  : {:.2f}'.format(model_name, f1_score(best_model_pred, test_labels)))
    
    confusion_mat = confusion_matrix(test_labels, best_model_pred, labels=[1, 0])
    print('This is the confusion matrix:')
    plt.figure(num=f'Confusion Matrix A1 {model_name}')
    plt.title(f'Confusion Matrix A1 {model_name}')
    ax = sns.heatmap((confusion_mat / np.sum(confusion_mat)*100),
                annot=True, fmt=".2f", cmap="crest")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()