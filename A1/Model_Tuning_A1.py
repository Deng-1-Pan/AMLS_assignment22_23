import numpy as np
import A1.landmarks as lm

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Read the data
features_train, train_labels = lm.extract_features_labels("Train")
features_test, test_labels = lm.extract_features_labels("Test")
# Reshape the data to fit the model
A1_train = features_train.reshape((features_train.shape[0], features_train.shape[1]*features_train.shape[2]))
A1_train_labels = train_labels[0]
A1_test = features_test.reshape((features_test.shape[0], features_test.shape[1]*features_test.shape[2]))
A1_test_labels = test_labels[0]
# Normalization
scaler = StandardScaler()
A1_train_data = scaler.fit_transform(A1_train)
A1_test_data = scaler.fit_transform(A1_test)
# Split Train and Validation
A1_train_data, A1_Validation_data, A1_train_labels, A1_Validation_labdels = train_test_split(A1_train_data, A1_train_labels, test_size=0.3, random_state=0)

# Finding the best model given the parameters
classifiers = [svm.LinearSVC(dual=False, random_state=0, fit_intercept = False),
               svm.SVC(random_state = 0),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state=0),
               AdaBoostClassifier(random_state=0)]
parameter_spaces = [{'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2]} ,
                    {'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2], 
                    	'degree' : [3, 4, 5, 6],
                                  'kernel': ['rbf', 'poly']},
                    {'n_neighbors': list(range(1, int(np.rint(np.sqrt(len(A1_train))))))},
                    {'n_estimators': list(range(100, 1001, 100))} ,
                    {'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]}]

for i in range(len(classifiers)):
   classifier = classifiers[i]
   print('Tuning on model' + str(classifier))
   tuning_model = GridSearchCV(classifier, parameter_spaces[i], scoring = 'accuracy', n_jobs=-1, cv = 5)
   tuning_model.fit(A1_train_data, A1_train_labels)
   print('Best parameters found for:\n' + str(classifier), tuning_model.best_params_)
   
   acc_train = accuracy_score(A1_train_labels, tuning_model.predict(A1_train_data))
   acc_val = accuracy_score(A1_Validation_labdels, tuning_model.predict(A1_Validation_data))
   acc_test = accuracy_score(A1_test_labels, tuning_model.predict(A1_test_data))
   print('For ' + classifiers[i] + ' the training Accuracy :', acc_train)
   print('For ' + classifiers[i] + ' the validation Accuracy :', acc_val)
   print('For ' + classifiers[i] + ' the test Accuracy :', acc_test)
   print('classification_report on the test set:')
   print(classification_report(A1_test_labels, tuning_model.predict(A1_test_data)))



# ============================ LinearSVC ======================================
# Best parameters found:
#  {'C': 0.001}
# For Linear SVC the training Accuracy : 0.9264004767580453
# For Linear SVC the validation Accuracy : 0.9159138290479499
# For Linear SVC the test Accuracy : 0.9143446852425181

# =============================== SVM =========================================
# Best parameters found:
# {'C': 0.001, 'degree': 6, 'kernel': 'poly'}
# For SVM the training Accuracy : 0.930870083432658
# For SVM the validation Accuracy : 0.9200833912439194
# For SVM the test Accuracy : 0.9122807017543859

# # =============================== KNN =========================================
# Best parameters found: KNeighborsClassifier()
# {'n_neighbors': 33}
# For Linear SVC the training Accuracy : 0.8218116805721096
# For Linear SVC the validation Accuracy : 0.8082001389854065
# For Linear SVC the test Accuracy : 0.782249742002064
# Best parameters found: KNeighborsClassifier(weights = 'distance')
#  {'n_neighbors': 30}
# For KNN the training Accuracy : 1.0
# For KNN the validation Accuracy : 0.8255733148019458
# For KNN the test Accuracy : 0.8245614035087719


# Best parameters found: KNeighborsClassifier(p = 1)
#  {'n_neighbors': 29}
# For KNN the training Accuracy : 0.8370083432657927
# For KNN the validation Accuracy : 0.8234885337039611
# For KNN the test Accuracy : 0.8121775025799793
# Best parameters found: KNeighborsClassifier(weights = 'distance', p = 1)
#  {'n_neighbors': 68} 
# For KNN the training Accuracy : 1.0
# For KNN the validation Accuracy : 0.8262682418346073
# For KNN the test Accuracy : 0.8111455108359134

 
# ================================ RF =========================================
# Best parameters found:
#  {'n_estimators': 100}
# For KNN the training Accuracy : 1.0
# For KNN the validation Accuracy : 0.8735232800555942
# For KNN the test Accuracy : 0.8524251805985552

# ============================== AdaBoost =====================================
# Best parameters found:
#  {'learning_rate': 0.5}
# For Linear SVC the training Accuracy : 0.8775327771156138
# For Linear SVC the validation Accuracy : 0.8742182070882557
# For Linear SVC the test Accuracy : 0.8544891640866873

# =============================== MLP =========================================
# Best parameters found:
#  {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (130,)} 2000 can converge
# For KNN the training Accuracy : 1.0
# For KNN the validation Accuracy : 0.9075747046560111
# For KNN the test Accuracy : 0.9029927760577915

# activation='tanh', solver = 'sgd', alpha=0.0001, hidden_layer_sizes = (130,), random_state=0, max_iter=5000
# For KNN the training Accuracy : 0.9713945172824792
# For KNN the validation Accuracy : 0.9200833912439194
# For KNN the test Accuracy : 0.9153766769865841
