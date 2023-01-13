import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

def Model_Training_Testing_B1(train_data, train_labels, test_data, test_labels):
    # Define the model
    model = MLPClassifier(activation = 'tanh', alpha= 0.05, hidden_layer_sizes =(140,), learning_rate = 'adaptive', solver = 'adam',random_state=3, max_iter=10000)
    
    # Normalize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)
    

    # Training the model
    with tqdm(desc="Training") as pbar:
        model.fit(train_data, train_labels)
        pbar.update(1)


    # Inference stage
    pred_test = model.predict(test_data)
    
    precision = precision_score(pred_test, test_labels, average = 'macro')
    recall = recall_score(pred_test, test_labels, average = 'macro')
    accuracy = accuracy_score(test_labels, pred_test)
    f1 = f1_score(pred_test, test_labels, average = 'macro')
    
    print(f'Precision on test dataset for MLP Classifier: {precision:.4f}')
    print(f'Recall on test dataset for MLP Classifier: {recall:.4f}')
    print(f'Accuracy on test dataset for MLP Classifier: {accuracy:.4f}')
    print(f'F1 score on test dataset for MLP Classifier: {f1:.4f}\n')
    
    confusion_mat = confusion_matrix(test_labels, pred_test, labels=model.classes_)
    print('This is the confusion matrix:')
    plt.figure(num='Confusion Matrix B1 MLP Classifier')
    plt.title('Confusion Matrix B1 MLP Classifier')
    ax = sns.heatmap((confusion_mat / np.sum(confusion_mat)*100),
                annot=True, fmt=".2f", cmap="crest")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()