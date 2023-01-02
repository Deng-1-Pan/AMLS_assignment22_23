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
A1_train_data = features_train['A1'].reshape(
    (features_train['A1'].shape[0], features_train['A1'].shape[1]*features_train['A1'].shape[2]))
A1_train_labels = train_labels[0]
A1_test_data = features_test['A1'].reshape(
    (features_test['A1'].shape[0], features_test['A1'].shape[1]*features_test['A1'].shape[2]))
A1_test_labels = test_labels[0]
# Normalization
scaler = StandardScaler()
A1_train_data = scaler.fit_transform(A1_train_data)
A1_test_data = scaler.fit_transform(A1_test_data)
# Split Train and Validation
A1_train_data, A1_Validation_data, A1_train_labels, A1_Validation_labdels = train_test_split(
    A1_train_data, A1_train_labels, test_size=0.3, random_state=0)


# A2_train_data = features_train['A2'].reshape((features_train['A2'].shape[0], features_train['A2'].shape[1]*features_train['A2'].shape[2]))
# A2_train_labels = train_labels[1]
# A2_test_data = features_test['A2'].reshape((features_test['A2'].shape[0], features_test['A2'].shape[1]*features_test['A2'].shape[2]))
# A2_test_labels = test_labels[1]
# # Normalization
# scaler = StandardScaler()
# A2_train_data = scaler.fit_transform(A2_train_data)
# A2_test_data = scaler.fit_transform(A2_test_data)
# # Split Train and Validation
# A2_train_data, A2_Validation_data, A2_train_labels, A2_Validation_labdels = train_test_split(A2_train_data, A2_train_labels, test_size=0.3, random_state=1)

classifiers = [svm.LinearSVC(dual=False, random_state=0, fit_intercept=False),
               svm.SVC(random_state=0),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state=0),
               AdaBoostClassifier(random_state=0)]
parameter_spaces = [{'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2]},
                    {'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2],
                    	'degree': [3, 4, 5, 6, 7, 8],
                     'kernel': ['rbf', 'poly']},
                    {'n_neighbors': list(
                        range(1, int(np.rint(np.sqrt(len(A1_train_data))))))},
                    {'n_estimators': list(range(100, 501, 100))},
                    {'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]}]

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
   acc_val = accuracy_score(A1_Validation_labdels,
                            tuning_model.predict(A1_Validation_data))
   acc_test = accuracy_score(
       A1_test_labels, tuning_model.predict(A1_test_data))
   print('For Linear SVC the training Accuracy :', acc_train)
   print('For Linear SVC the validation Accuracy :', acc_val)
   print('For Linear SVC the test Accuracy :', acc_test)
   print('classification_report on the test set:')
   print(classification_report(A1_test_labels, tuning_model.predict(A1_test_data)))


# # ============================ LinearSVC ======================================
# classifier = svm.LinearSVC(dual=False, random_state=0, fit_intercept = False)
# parameter_space = {'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2]}
# model = GridSearchCV(classifier, parameter_space, scoring = 'accuracy', n_jobs=-1, cv=5)
# model.fit(A1_train_data, A1_train_labels)
# print('Best parameters found:\n', model.best_params_)

# acc_train = accuracy_score(A1_train_labels, model.predict(A1_train_data))
# acc_val = accuracy_score(A1_Validation_labdels, model.predict(A1_Validation_data))
# acc_test = accuracy_score(A1_test_labels, model.predict(A1_test_data))
# print('For Linear SVC the training Accuracy :', acc_train)
# print('For Linear SVC the validation Accuracy :', acc_val)
# print('For Linear SVC the test Accuracy :', acc_test)
# print('classification_report on the test set:')
# print(classification_report(A1_test_labels, model.predict(A1_test_data)))
# # Best parameters found:
# #  {'C': 0.001}
# # For Linear SVC the training Accuracy : 0.9264004767580453
# # For Linear SVC the validation Accuracy : 0.9159138290479499
# # For Linear SVC the test Accuracy : 0.9143446852425181

# # =============================== SVM =========================================
# clf = svm.SVC(random_state = 0)
# parameter_space = {'C': [1e-3, 1e-2, 0.1, 1, 10, 1e2],
# 	'degree' : [3, 4, 5, 6 ,7 ,8],
#               'kernel': ['rbf', 'poly']}
# model = GridSearchCV(clf, parameter_space, scoring = 'accuracy', n_jobs=-1, cv=5)
# model.fit( A1_train_data, A1_train_labels )
# print('Best parameters found:\n', model.best_params_)

# acc_train = accuracy_score(A1_train_labels, model.predict(A1_train_data))
# acc_val = accuracy_score(A1_Validation_labdels, model.predict(A1_Validation_data))
# acc_test = accuracy_score(A1_test_labels, model.predict(A1_test_data))
# print('For Linear SVC the training Accuracy :', acc_train)
# print('For Linear SVC the validation Accuracy :', acc_val)
# print('For Linear SVC the test Accuracy :', acc_test)
# print('classification_report on the test set:')
# print(classification_report(A1_test_labels, model.predict(A1_test_data)))
# # Best parameters found:
# # {'C': 0.001, 'degree': 6, 'kernel': 'poly'}
# # For SVM the training Accuracy : 0.930870083432658
# # For SVM the validation Accuracy : 0.9200833912439194
# # For SVM the test Accuracy : 0.9122807017543859

# # =============================== KNN =========================================
# clf = KNeighborsClassifier()
# parameter_space = {'n_neighbors': list(range(1, int(np.rint(np.sqrt(len(A1_train_data))))))}
# model = GridSearchCV(clf, parameter_space, scoring = 'accuracy', n_jobs=-1, cv=5)
# model.fit( A1_train_data, A1_train_labels )
# print('Best parameters found:\n', model.best_params_)

# acc_train = accuracy_score(A1_train_labels, model.predict(A1_train_data))
# acc_val = accuracy_score(A1_Validation_labdels, model.predict(A1_Validation_data))
# acc_test = accuracy_score(A1_test_labels, model.predict(A1_test_data))
# print('For KNN the training Accuracy :', acc_train)
# print('For KNN the validation Accuracy :', acc_val)
# print('For KNN the test Accuracy :', acc_test)
# print('classification_report on the test set:')
# print(classification_report(A1_test_labels, model.predict(A1_test_data)))
# # Best parameters found:
# # {'n_neighbors': 33}
# # For Linear SVC the training Accuracy : 0.8218116805721096
# # For Linear SVC the validation Accuracy : 0.8082001389854065
# # For Linear SVC the test Accuracy : 0.782249742002064

# # ================================ RF =========================================
# clf = RandomForestClassifier(random_state=0)
# parameter_space = {'n_estimators': list(range(100, 501, 100))}
# model = GridSearchCV(clf, parameter_space, scoring = 'accuracy', n_jobs=-1, cv=5)
# model.fit( A1_train_data, A1_train_labels )
# print('Best parameters found:\n', model.best_params_)

# acc_train = accuracy_score(A1_train_labels, model.predict(A1_train_data))
# acc_val = accuracy_score(A1_Validation_labdels, model.predict(A1_Validation_data))
# acc_test = accuracy_score(A1_test_labels, model.predict(A1_test_data))
# print('For Linear SVC the training Accuracy :', acc_train)
# print('For Linear SVC the validation Accuracy :', acc_val)
# print('For Linear SVC the test Accuracy :', acc_test)
# print('classification_report on the test set:')
# print(classification_report(A1_test_labels, model.predict(A1_test_data)))
# # Best parameters found:
# #  {'n_estimators': 200}
# # For RF the training Accuracy : 1.0
# # For RF the validation Accuracy : 0.8749131341209173
# # For RF the test Accuracy : 0.8534571723426213

# # ============================== AdaBoost =====================================
# clf = AdaBoostClassifier(random_state=0)
# parameter_space = {'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]}
# model = GridSearchCV(clf, parameter_space, scoring = 'accuracy', n_jobs=-1, cv=5)
# model.fit( A1_train_data, A1_train_labels )
# print('Best parameters found:\n', model.best_params_)

# y_true, y_pred = A1_test_labels, model.predict( A1_test_data )
# from sklearn.metrics import classification_report
# print('Results on the test set:')
# print(classification_report(y_true, y_pred))

# acc_train = accuracy_score(A1_train_labels, model.predict(A1_train_data))
# acc_val = accuracy_score(A1_Validation_labdels, model.predict(A1_Validation_data))
# acc_test = accuracy_score(A1_test_labels, model.predict(A1_test_data))
# print('For AdaBoostClassifier the training Accuracy :', acc_train)
# print('For AdaBoostClassifier the validation Accuracy :', acc_val)
# print('For AdaBoostClassifier the test Accuracy :', acc_test)
# print('classification_report on the test set:')
# print(classification_report(A1_test_labels, model.predict(A1_test_data)))
# # Best parameters found:
# #  {'learning_rate': 0.5}
# # For Linear SVC the training Accuracy : 0.8775327771156138
# # For Linear SVC the validation Accuracy : 0.8742182070882557
# # For Linear SVC the test Accuracy : 0.8544891640866873
