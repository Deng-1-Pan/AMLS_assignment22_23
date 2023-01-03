import numpy as np
import A1.landmarks as lm

from sklearn import svm
from sklearn.model_selection import train_test_split
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
# Split Train and Validation
A2_train_data, A2_Validation_data, A2_train_labels, A2_Validation_labdels = train_test_split(
    A2_train_data, A2_train_labels, test_size=0.3, random_state=1)

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
                    {'n_neighbors': list(range(1, 1 + int(np.rint(np.sqrt(len(A2_train_data)+len(A2_Validation_data)))))),
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
    acc_val = accuracy_score(A2_Validation_labdels,
                             tuning_model.predict(A2_Validation_data))
    acc_test = accuracy_score(
        A2_test_labels, tuning_model.predict(A2_test_data))
    print('For ' + classifiers[i] + ' the training Accuracy :', acc_train)
    print('For ' + classifiers[i] + ' the validation Accuracy :', acc_val)
    print('For ' + classifiers[i] + ' the test Accuracy :', acc_test)
    print('classification_report on the test set:')
    print(classification_report(A2_test_labels,
          tuning_model.predict(A2_test_data)))


# ============================ LinearSVC ======================================
# Best parameters found:
#  {'C': 0.01}
# For Linear SVC the training Accuracy : 0.8971990464839095
# For Linear SVC the validation Accuracy : 0.8922863099374566
# For Linear SVC the test Accuracy : 0.8864809081527347
# =============================== SVM =========================================
# Best parameters found: kernel = 'rbf'
#  {'C': 1}
# For Linear SVC the training Accuracy : 0.9207389749702026
# For Linear SVC the validation Accuracy : 0.8929812369701181
# For Linear SVC the test Accuracy : 0.9019607843137255
# Best parameters found: kernel = 'poly' C = 1
#  {'degree': 3}
# For Linear SVC the training Accuracy : 0.9112038140643623
# For Linear SVC the validation Accuracy : 0.8853370396108409
# For Linear SVC the test Accuracy : 0.8957688338493293
# =============================== KNN =========================================
# Best parameters found: KNeighborsClassifier()
#  {'n_neighbors': 27}
# For Linear SVC the training Accuracy : 0.8879618593563766
# For Linear SVC the validation Accuracy : 0.8832522585128562
# For Linear SVC the test Accuracy : 0.8926728586171311
# Best parameters found: KNeighborsClassifier(weights = 'distance')
#  {'n_neighbors': 27}
# For Linear SVC the training Accuracy : 1.0
# For Linear SVC the validation Accuracy : 0.8839471855455178
# For Linear SVC the test Accuracy : 0.8926728586171311

# Best parameters found: KNeighborsClassifier(p = 1)
#  {'n_neighbors': 39}
# For Linear SVC the training Accuracy : 0.8885578069129917
# For Linear SVC the validation Accuracy : 0.8874218207088256
# For Linear SVC the test Accuracy : 0.8988648090815273
# Best parameters found: KNeighborsClassifier(weights = 'distance', p = 1)
#  {'n_neighbors': 40}
# For Linear SVC the training Accuracy : 1.0
# For Linear SVC the validation Accuracy : 0.8853370396108409
# For Linear SVC the test Accuracy : 0.8968008255933952

# parameter_space = {'n_neighbors': list(range(1, int(np.rint(np.sqrt(len(A2_train_data)))))),
#                     'weights': ['uniform', 'distance'],
#                     'p' : [1, 2]}
# Best parameters found:
#  {'n_neighbors': 40, 'p': 1, 'weights': 'distance'}
# For Linear SVC the training Accuracy : 1.0
# For Linear SVC the validation Accuracy : 0.8853370396108409
# For Linear SVC the test Accuracy : 0.8968008255933952

# ================================ RF =========================================
# Best parameters found:
#  {'n_estimators': 900}
# For Linear SVC the training Accuracy : 1.0
# For Linear SVC the validation Accuracy : 0.8908964558721334
# For Linear SVC the test Accuracy : 0.9040247678018576

# ============================== AdaBoost =====================================
# Best parameters found:
#  {'learning_rate': 0.5}
# For Linear SVC the training Accuracy : 0.8957091775923719
# For Linear SVC the validation Accuracy : 0.8895066018068103
# For Linear SVC the test Accuracy : 0.8988648090815273

# =============================== MLP =========================================
# Random_seed = 3
# activation='tanh', solver = 'sgd', learning_rate = 'adaptive', alpha=0.0001, hidden_layer_sizes = (130,)
# For KNN the training Accuracy : 0.9696066746126341
# For KNN the validation Accuracy : 0.8846421125781793
# For KNN the test Accuracy : 0.891640866873065

# activation='tanh', solver = 'sgd', alpha=0.0001, hidden_layer_sizes = (130,)
# For KNN the training Accuracy : 0.9705005959475567
# For KNN the validation Accuracy : 0.8846421125781793
# For KNN the test Accuracy : 0.890608875128999

# activation='tanh', alpha=0.0001, hidden_layer_sizes = (130,)
# For KNN the training Accuracy : 1.0
# For KNN the validation Accuracy : 0.8776928422515636
# For KNN the test Accuracy : 0.8802889576883385

# solver = 'sgd', learning_rate = 'adaptive', alpha=0.0001, hidden_layer_sizes = (130,)
# For KNN the training Accuracy : 0.9749702026221693
# For KNN the validation Accuracy : 0.8902015288394719
# For KNN the test Accuracy : 0.8988648090815273

# solver = 'sgd', alpha=0.05, hidden_layer_sizes = (200,)
# For KNN the training Accuracy : 0.9716924910607867
# For KNN the validation Accuracy : 0.8922863099374566
# For KNN the test Accuracy : 0.8895768833849329

#  {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (200,)}
# For KNN the training Accuracy : 1.0
# For KNN the validation Accuracy : 0.8902015288394719
# For KNN the test Accuracy : 0.8823529411764706

# Best parameters found:
#  {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (100,), 'learning_rate': 'adaptive', 'solver': 'sgd'}
# For KNN the training Accuracy : 0.9436829558998808
# For KNN the validation Accuracy : 0.8929812369701181
# For KNN the test Accuracy : 0.8978328173374613
