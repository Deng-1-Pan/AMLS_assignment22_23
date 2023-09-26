# Machine Learning Approaches to Feature Recognition: Insights from Experiments on CelebA and Cartoon Set Datasets

## Abstract
This report presents the results of an experiment exploring the performance of different machine learning models on the celeba and cartoon\_set datasets. Five different methods, including Support Vector Machine (SVM), K-Nearest Neighbour (KNN), Random Forest, Adaboost and Multi-Layer Perceptron (MLP), were used to test the two datasets in different ways. For the celeba dataset, the detection of gender and the detection of smiles were tested, while for the cartoon\_set dataset, face shape discrimination and eye colour recognition were tested. The experimental results show that MLP performs well on the celeba dataset for gender and smile detection. While for the cartoon\_set dataset, MLP performed well in face shape recognition, while KNN performed well in eye colour recognition. The report discusses and analyses the results of the relevant model runs and possible directions for tuning. This report also discusses the use of pre-processing steps and the evaluation of the models using evaluation metrics such as precision, recall, accuracy and F1 scores. This experiment highlights the importance of selecting the right machine-learning model for a particular task, dataset, and feature extraction. The relevant code for this report is available on GitHub

## Requirement
- In this report, all coding is done in Python, which uses version 3.9.15. And the following packages were used:
    - numpy 1.21.5
    - seaborn 0.11.2
    - matplotlib 3.5.1
    - tqdm 4.64.1
    - sklearn 1.0.2
    - cv2 4.6.0
    - dlib 19.24.0
    - keras_preprocessing 1.1.2

## File Description
There are 4 different task files that hold py files to solve different tasks. the database folder does not hold any data, so that users can put in the data they need to use. main.py is the repository execution file.

### A1
1. landmarks.py: Used to extract feature
2. Model_Tuning_A1.py: Used for testing, hyperparameter tuning and comparing files for different models
3. shape_predictor_68_face_landmarks.dat: pre-trained model for extracting facial features
4. Traing_Models_A1.py: Pre-trained model to solve task A1

### A2
1. Model_Tuning_A2.py: Used for testing, hyperparameter tuning and comparing files for different models
2. Traing_Models_A2.py: Pre-trained model to solve task A2

### B1
1. featire_extractopm.py: Used to extract feature
2. Model_Tuning_B1.py: Used for testing, hyperparameter tuning and comparing files for different models
3. Traing_Models_B1.py: Pre-trained model to solve task B1

### B2
1. Model_Tuning_B2.py: Used for testing, hyperparameter tuning and comparing files for different models
2. Traing_Models_A2.py: Pre-trained model to solve task B2
