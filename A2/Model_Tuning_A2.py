import numpy as np
import A1.landmarks as lm
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



# Read the data
features_train, train_labels = lm.extract_features_labels("Train")
features_test, test_labels = lm.extract_features_labels("Test")

# Reshape the data to fit the model
A2_train = features_train.reshape(
    (features_train.shape[0], features_train.shape[1]*features_train.shape[2]))
A2_test = features_test.reshape(
    (features_test.shape[0], features_test.shape[1]*features_test.shape[2]))
A2_train_labels = train_labels[1]
A2_test_labels = test_labels[1]
# Normalization
scaler = StandardScaler()
A2_train_data = scaler.fit_transform(A2_train)
A2_test_data = scaler.fit_transform(A2_test)

classifiers = [svm.LinearSVC(dual=False, random_state=0, fit_intercept=False),
               svm.SVC(random_state=0),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state=0),
               AdaBoostClassifier(random_state=0),
               MLPClassifier(random_state=3, max_iter=5000)]
parameter_spaces = [{'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2]},
                    {'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2],
                        'degree': [3, 4, 5, 6],
                        'kernel': ['rbf', 'poly']},
                    {'n_neighbors': list(range(1, 1 + int(np.rint(np.sqrt(len(A2_train_data)))))),
                     'weights': ['uniform', 'distance'],
                     'p': [1, 2]},
                    {'n_estimators': list(range(100, 1001, 100))},
                    {'learning_rate': [1e-4, 5e-4, 1e-3,
                                       5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]},
                    {'hidden_layer_sizes': [(x,) for x in list(range(100, 201, 10))],
                     'activation': ['tanh', 'relu'],
                     'alpha': [0.0001, 0.05],
                     'learning_rate': ['adaptive', 'constant'],
                     'solver': ['sgd', 'adam']}]

for i in range(len(classifiers)):
    classifier = classifiers[i]
    print('Tuning on model ' + str(classifier))
    tuning_model = GridSearchCV(
        classifier, parameter_spaces[i], scoring='accuracy', n_jobs=-1, cv=5)
    tuning_model.fit(A2_train_data, A2_train_labels)
    print('Best parameters found for:\n' +
          str(classifier), tuning_model.best_params_)

    acc_train = accuracy_score(
        A2_train_labels, tuning_model.predict(A2_train_data))
    acc_test = accuracy_score(
        A2_test_labels, tuning_model.predict(A2_test_data))
    print('For ' + classifiers[i] + ' the training Accuracy :', acc_train)
    print('For ' + classifiers[i] + ' the test Accuracy :', acc_test)
    print('classification_report on the test set:')
    print(classification_report(A2_test_labels,
          tuning_model.predict(A2_test_data)))
    
# Visualisation training and validation learning curve
clf_svc = svm.LinearSVC(C = 0.001, dual=False, random_state=0, fit_intercept = False)
clf_svm = svm.SVC(C = 1, kernel = 'rbf', random_state=0)
clf_knn = KNeighborsClassifier(n_neighbors = 40, p = 1, weights = 'distance')
clf_rf = RandomForestClassifier(n_estimators = 400, random_state=0)
clf_ada = AdaBoostClassifier(learning_rate = 1, random_state=0)
clf_mlp = MLPClassifier(activation = 'relu', alpha= 0.05, hidden_layer_sizes =(140,), learning_rate = 'adaptive', solver = 'sgd',random_state=3, max_iter=10000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2))
plt.suptitle("Learning Curves")
for i in [clf_svc, clf_svm, clf_knn, clf_rf, clf_ada, clf_mlp]:
    train_sizes, train_scores, val_scores = learning_curve(i, A2_train_data, A2_train_labels, cv=5, n_jobs = -1)
    ax1.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label= str(type(i)).split('.')[-1].split("'")[0] +" Training Score")
ax1.set_xlabel("Training Set Size")
ax1.set_ylabel("Accuracy Score")
ax1.legend(loc="best", prop = {'size':8})
ax1.grid()
for i in [clf_svc, clf_svm, clf_knn, clf_rf, clf_ada, clf_mlp]:
    train_sizes, train_scores, val_scores = learning_curve(i, A2_train_data, A2_train_labels, cv=5, n_jobs = -1)
    ax2.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label= str(type(i)).split('.')[-1].split("'")[0] + " Validation Score")
ax2.set_xlabel("Training Set Size")
ax2.set_ylabel("Accuracy Score")
ax2.legend(loc="best", prop = {'size':8})
ax2.grid()
plt.savefig('./A2/learning_curve_A2.png')


# ============================ LinearSVC ======================================
# LinearSVC(dual=False, fit_intercept=False, random_state=0) {'C': 0.001}
# For LinearSVC(dual=False, fit_intercept=False, random_state=0) the training Accuracy : 0.8886339937434828
# For LinearSVC(dual=False, fit_intercept=False, random_state=0) the test Accuracy : 0.8875128998968008

# =============================== SVM =========================================
# SVC(random_state=0) {'C': 1, 'degree': 3, 'kernel': 'rbf'}
# For SVC(random_state=0) the training Accuracy : 0.9226277372262773
# For SVC(random_state=0) the test Accuracy : 0.8978328173374613

# =============================== KNN =========================================
# KNeighborsClassifier() {'n_neighbors': 40, 'p': 1, 'weights': 'distance'}
# For KNeighborsClassifier() the training Accuracy : 1.0
# For KNeighborsClassifier() the test Accuracy : 0.8968008255933952

# ================================ RF =========================================
# RandomForestClassifier(random_state=0) {'n_estimators': 400}
# For RandomForestClassifier(random_state=0) the training Accuracy : 1.0
# For RandomForestClassifier(random_state=0) the test Accuracy : 0.8957688338493293

# ============================== AdaBoost =====================================
# AdaBoostClassifier(random_state=0) {'learning_rate': 1}
# For AdaBoostClassifier(random_state=0) the training Accuracy : 0.8986444212721585
# For AdaBoostClassifier(random_state=0) the test Accuracy : 0.890608875128999

# =============================== MLP =========================================
# MLPClassifier(max_iter=5000, random_state=3) {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (140,), 'learning_rate': 'adaptive', 'solver': 'sgd'}
# For MLPClassifier(max_iter=5000, random_state=3) the training Accuracy : 0.9705943691345151
# For MLPClassifier(max_iter=5000, random_state=3) the test Accuracy : 0.8885448916408669