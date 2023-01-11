import pickle
import numpy as np
import B1.feature_extraction as fx

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
features_train, train_labels = fx.extract_features_labels("Train")
features_test, test_labels = fx.extract_features_labels("Test")

# Reshape the data to fit the model
B1_train = features_train['Face'].reshape((features_train['Face'].shape[0], features_train['Face'].shape[1]*features_train['Face'].shape[2]))
B1_train_labels = train_labels[0]
B1_test = features_test['Face'].reshape((features_test['Face'].shape[0], features_test['Face'].shape[1]*features_test['Face'].shape[2]))
B1_test_labels = test_labels[0]

# Normalization
scaler = StandardScaler()
B1_train_data = scaler.fit_transform(B1_train)
B1_test_data = scaler.fit_transform(B1_test)

# Split Train and Validation
B1_train_data, B1_Validation_data, B1_train_labels, B1_Validation_labdels = train_test_split(B1_train_data, B1_train_labels, test_size=0.3, random_state=2)

# Finding the best model given the parameters
classifiers = [svm.LinearSVC(dual=False, random_state=1, fit_intercept = False),
               svm.SVC(random_state = 1),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state=1),
               AdaBoostClassifier(random_state=1),
               MLPClassifier(random_state=3, max_iter=10000)]
parameter_spaces = [{'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2]} ,
                    {'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2], 
                     'degree' : [3, 4, 5, 6],
                     'kernel': ['rbf', 'poly'],
                     'max_iter': [100000] },
                    {'n_neighbors': list(range(1, 1 + int(np.rint(np.sqrt(len(B1_train_data)+len(B1_Validation_data)))))),
                     'weights': ['uniform', 'distance'],
                     'p' : [1, 2]},
                    {'n_estimators': list(range(100, 1001, 100))} ,
                    {'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]},
                    {'hidden_layer_sizes': [ (x,) for x in list(range(100, 201, 10))],
                     'activation': ['tanh', 'relu'],
                     'alpha': [0.0001, 0.05],
                     'learning_rate' : ['adaptive', 'constant'],
                     'solver' : ['sgd', 'adam']}]

for i in range(len(classifiers)):
   classifier = classifiers[i]
   print('Tuning on model ' + str(classifier))
   tuning_model = GridSearchCV(classifier, parameter_spaces[i], scoring = 'accuracy', n_jobs=-1, cv = 5)
   tuning_model.fit(B1_train_data, B1_train_labels)
   print('Best parameters found for:\n' + str(classifier), tuning_model.best_params_)
   
   acc_train = accuracy_score(B1_train_labels, tuning_model.predict(B1_train_data))
   acc_val = accuracy_score(B1_Validation_labdels, tuning_model.predict(B1_Validation_data))
   acc_test = accuracy_score(B1_test_labels, tuning_model.predict(B1_test_data))
   print('For ' + str(classifier) + ' the training Accuracy :', acc_train)
   print('For ' + str(classifier) + ' the validation Accuracy :', acc_val)
   print('For ' + str(classifier) + ' the test Accuracy :', acc_test)
   print('classification_report on the test set:')
   print(classification_report(B1_test_labels, tuning_model.predict(B1_test_data)))

# ============================ LinearSVC ======================================
# Best parameters found for:
# LinearSVC(dual=False, fit_intercept=False, random_state=1) {'C': 0.1}
# For the training Accuracy : 0.6434176111595467
# For the validation Accuracy : 0.6189507930052867
# For  the test Accuracy : 0.6320431161195492

# =============================== SVM =========================================
# Best parameters found for: Training Completed! Time used: 254.55s
# SVC(random_state=1) {'C': 10, 'degree': 3, 'kernel': 'rbf', 'max_iter': 1000000}
# # For SVC(random_state=1) the training Accuracy : 0.8809067131647776
# # For SVC(random_state=1) the validation Accuracy : 0.7519316795445303
# # For SVC(random_state=1) the test Accuracy : 0.7604115629593337

# =============================== KNN =========================================
# Best parameters found:
# KNeighborsClassifier() {'n_neighbors': 7, 'p': 1, 'weights': 'distance'}
# the training Accuracy : 1.0
# the validation Accuracy : 0.6905246034973567
# the test Accuracy : 0.6761391474767271

# ================================ RF =========================================
# Best parameters found for:
# RandomForestClassifier(random_state=1) {'n_estimators': 600}
# For RandomForestClassifier(random_state=1) the training Accuracy : 1.0
# For RandomForestClassifier(random_state=1) the validation Accuracy : 0.7311915412769419
# For RandomForestClassifier(random_state=1) the test Accuracy : 0.7344439000489956

# ============================== AdaBoost =====================================
# Best parameters found for:
# AdaBoostClassifier(random_state=1) {'learning_rate': 1}
# For AdaBoostClassifier(random_state=1) the training Accuracy : 0.610462074978204
# For AdaBoostClassifier(random_state=1) the validation Accuracy : 0.6067507116714111
# For AdaBoostClassifier(random_state=1) the test Accuracy : 0.5987261146496815

# =============================== MLP =========================================
# Random_seed = 3
# Training Completed! Time used: 12622.64s 3.5 hours
# MLPClassifier(max_iter=10000, random_state=3) {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (180,), 'learning_rate': 'adaptive', 'solver': 'sgd'}
# For MLPClassifier(max_iter=10000, random_state=3) the training Accuracy : 0.8348735832606801
# For MLPClassifier(max_iter=10000, random_state=3) the validation Accuracy : 0.7503050020333469
# For MLPClassifier(max_iter=10000, random_state=3) the test Accuracy : 0.7633512983831455
# Training Completed! Time used: 77.01s for this model

