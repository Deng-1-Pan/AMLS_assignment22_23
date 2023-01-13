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
A1_train = features_train.reshape(
    (features_train.shape[0], features_train.shape[1]*features_train.shape[2]))
A1_train_labels = train_labels[0]
A1_test = features_test.reshape(
    (features_test.shape[0], features_test.shape[1]*features_test.shape[2]))
A1_test_labels = test_labels[0]
# Normalization
scaler = StandardScaler()
A1_train_data = scaler.fit_transform(A1_train)
A1_test_data = scaler.fit_transform(A1_test)

# Finding the best model given the parameters
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
                    {'n_neighbors': list(
                        range(1, int(np.rint(np.sqrt(len(A1_train))))))},
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
    print('Tuning on model' + str(classifier))
    tuning_model = GridSearchCV(
        classifier, parameter_spaces[i], scoring='accuracy', n_jobs=-1, cv=5)
    tuning_model.fit(A1_train_data, A1_train_labels)
    print('Best parameters found for:\n' +
          str(classifier), tuning_model.best_params_)

    acc_train = accuracy_score(
        A1_train_labels, tuning_model.predict(A1_train_data))
    acc_test = accuracy_score(
        A1_test_labels, tuning_model.predict(A1_test_data))
    print('For ' + classifiers[i] + ' the training Accuracy :', acc_train)
    print('For ' + classifiers[i] + ' the test Accuracy :', acc_test)
    print('classification_report on the test set:')
    print(classification_report(A1_test_labels,
          tuning_model.predict(A1_test_data)))

# Visualisation training and validation learning curve
clf_svc = svm.LinearSVC(C=0.1, dual=False, random_state=0, fit_intercept=False)
clf_svm = svm.SVC(C=10, kernel='rbf', random_state=0)
clf_knn = KNeighborsClassifier(n_neighbors=16, p=2, weights='distance')
clf_rf = RandomForestClassifier(n_estimators=700, random_state=0)
clf_ada = AdaBoostClassifier(learning_rate=1, random_state=0)
clf_mlp = MLPClassifier(activation='tanh', alpha=.05, hidden_layer_sizes=(
    180,), learning_rate='adaptive', solver='sgd', random_state=3, max_iter=10000)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2))
plt.suptitle("Learning Curves")
for i in [clf_svc, clf_svm, clf_knn, clf_rf, clf_ada, clf_mlp]:
    train_sizes, train_scores, val_scores = learning_curve(
        i, A1_train_data, A1_train_labels, cv=5, n_jobs=-1)
    ax1.plot(train_sizes, np.mean(train_scores, axis=1), 'o-',
             label=str(type(i)).split('.')[-1].split("'")[0] + " Training Score")
ax1.set_xlabel("Training Set Size")
ax1.set_ylabel("Accuracy Score")
ax1.legend(loc="best", prop={'size': 8})
ax1.grid()
for i in [clf_svc, clf_svm, clf_knn, clf_rf, clf_ada, clf_mlp]:
    train_sizes, train_scores, val_scores = learning_curve(
        i, A1_train_data, A1_train_labels, cv=5, n_jobs=-1)
    ax2.plot(train_sizes, np.mean(val_scores, axis=1), 'o-',
             label=str(type(i)).split('.')[-1].split("'")[0] + " Validation Score")
ax2.set_xlabel("Training Set Size")
ax2.set_ylabel("Accuracy Score")
ax2.legend(loc="best", prop={'size': 8})
ax2.grid()
plt.savefig('./A1/learning_curve_A1.png')


# ============================ LinearSVC ======================================
# LinearSVC(dual=False, fit_intercept=False, random_state=0) {'C': 0.1}
# For LinearSVC(dual=False, fit_intercept=False, random_state=0) the training Accuracy : 0.9334723670490094
# For LinearSVC(dual=False, fit_intercept=False, random_state=0) the test Accuracy : 0.9081527347781218

# =============================== SVM =========================================
# SVC(random_state=0) {'C': 10, 'degree': 3, 'kernel': 'rbf'}
# For SVC(random_state=0) the training Accuracy : 0.981021897810219
# For SVC(random_state=0) the test Accuracy : 0.9102167182662538

# =============================== KNN =========================================
# KNeighborsClassifier() {'n_neighbors': 16, 'p': 2, 'weights': 'distance'}
# For KNeighborsClassifier() the training Accuracy : 1.0
# For KNeighborsClassifier() the test Accuracy : 0.8194014447884417

# ================================ RF =========================================
# RandomForestClassifier(random_state=0) {'n_estimators': 700}
# For RandomForestClassifier(random_state=0) the training Accuracy : 1.0
# For RandomForestClassifier(random_state=0) the test Accuracy : 0.8565531475748194

# ============================== AdaBoost =====================================
# AdaBoostClassifier(random_state=0) {'learning_rate': 1}
# For AdaBoostClassifier(random_state=0) the training Accuracy : 0.8905109489051095
# For AdaBoostClassifier(random_state=0) the test Accuracy : 0.8637770897832817

# =============================== MLP =========================================
# MLPClassifier(max_iter=5000, random_state=3) {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (180,), 'learning_rate': 'adaptive', 'solver': 'sgd'}
# For MLPClassifier(max_iter=5000, random_state=3) the training Accuracy : 0.959332638164755
# For MLPClassifier(max_iter=5000, random_state=3) the test Accuracy : 0.9122807017543859