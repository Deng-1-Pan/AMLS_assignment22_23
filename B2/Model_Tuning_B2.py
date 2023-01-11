import cv2
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


def extract_features(image):
  # Extract the R, G, and B channels
  r = image[:,:,0]
  g = image[:,:,1]
  b = image[:,:,2]

  # Compute the histograms of the R, G, and B channels
  r_hist = cv2.calcHist([r], [0], None, [256], [0,256])
  g_hist = cv2.calcHist([g], [0], None, [256], [0,256])
  b_hist = cv2.calcHist([b], [0], None, [256], [0,256])

  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((r_hist, g_hist, b_hist))

  return hist_features.flatten()

# Read the data
features_train, train_labels = fx.extract_features_labels("Train")
features_test, test_labels = fx.extract_features_labels("Test")

B2_train =  np.array([extract_features(image) for image in features_train['Eyes']])
B2_test = np.array([extract_features(image) for image in features_test['Eyes']])
B2_train_labels = train_labels[1]
B2_test_labels = test_labels[1]

# Normalization
scaler = StandardScaler()
B2_train_data = scaler.fit_transform(B2_train)
B2_test_data = scaler.fit_transform(B2_test)

# Split Train and Validation
B2_train_data, B2_Validation_data, B2_train_labels, B2_Validation_labdels = train_test_split(B2_train_data, B2_train_labels, test_size=0.3, random_state=1)

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
                    {'n_neighbors': list(range(1, 1 + int(np.rint(np.sqrt(len(B2_train_data)+len(B2_Validation_data)))))),
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
    tuning_model.fit(B2_train_data, B2_train_labels)
    print('Best parameters found for:\n' + str(classifier), tuning_model.best_params_)
        
    acc_train = accuracy_score(B2_train_labels, tuning_model.predict(B2_train_data))
    acc_val = accuracy_score(B2_Validation_labdels, tuning_model.predict(B2_Validation_data))
    acc_test = accuracy_score(B2_test_labels, tuning_model.predict(B2_test_data))
    print('For ' + str(classifier) + ' the training Accuracy :', acc_train)
    print('For ' + str(classifier) + ' the validation Accuracy :', acc_val)
    print('For ' + str(classifier) + ' the test Accuracy :', acc_test)
    print('classification_report on the test set:')
    print(classification_report(B2_test_labels, tuning_model.predict(B2_test_data)))

   
# ============================ LinearSVC ======================================
# Fail to converge
# Time elapsed: 26222.22 seconds approx 7 hours
# LinearSVC(dual=False, fit_intercept=False, random_state=1) {'C': 0.01, 'max_iter': 100000}
# For LinearSVC(dual=False, fit_intercept=False, random_state=1) the training Accuracy : 0.9102005231037489
# For LinearSVC(dual=False, fit_intercept=False, random_state=1) the validation Accuracy : 0.8666124440829606
# For LinearSVC(dual=False, fit_intercept=False, random_state=1) the test Accuracy : 0.8598726114649682

# =============================== SVM =========================================
# SVC(random_state=1) {'C': 100.0, 'degree': 3, 'kernel': 'poly', 'max_iter': 100000}
# For SVC(random_state=1) the training Accuracy : 0.9274629468177855
# For SVC(random_state=1) the validation Accuracy : 0.8698657991053274
# For SVC(random_state=1) the test Accuracy : 0.8535031847133758

# =============================== KNN =========================================
# Time elapsed: 1174.91 seconds
# Best parameters found for:
# KNeighborsClassifier() {'n_neighbors': 2, 'p': 1, 'weights': 'distance'}
# For KNeighborsClassifier() the training Accuracy : 0.9644289450741064
# For KNeighborsClassifier() the validation Accuracy : 0.8767791785278568
# For KNeighborsClassifier() the test Accuracy : 0.8745712885840274

# ================================ RF =========================================
# Time elapsed: 766.98 seconds
# Best parameters found for:
# RandomForestClassifier(random_state=1) {'n_estimators': 1700}
# For RandomForestClassifier(random_state=1) the training Accuracy : 0.9659982563208369
# For RandomForestClassifier(random_state=1) the validation Accuracy : 0.8743391622610818
# For RandomForestClassifier(random_state=1) the test Accuracy : 0.8682018618324351

# ============================== AdaBoost =====================================
# Time elapsed: 57.28 seconds
# Best parameters found for:
# AdaBoostClassifier(random_state=1) {'learning_rate': 0.1}
# For AdaBoostClassifier(random_state=1) the training Accuracy : 0.8741063644289451
# For AdaBoostClassifier(random_state=1) the validation Accuracy : 0.8649857665717772
# For AdaBoostClassifier(random_state=1) the test Accuracy : 0.8564429201371877

# =============================== MLP =========================================
# MLPClassifier(max_iter=10000, random_state=3) {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (150,), 'learning_rate': 'adaptive', 'solver': 'adam'}
# For MLPClassifier(max_iter=10000, random_state=3) the training Accuracy : 0.9295553618134264
# For MLPClassifier(max_iter=10000, random_state=3) the validation Accuracy : 0.8743391622610818
# For MLPClassifier(max_iter=10000, random_state=3) the test Accuracy : 0.8682018618324351
