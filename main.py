import pandas as pd
from importlib.machinery import SourceFileLoader

import src.landmarks as lm

def Load_data(type):
    if type == 'A':
        print("==============================Loading data for Task A==============================")
        # training_datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23/celeba/labels.csv', sep='\t', index_col=0)
        # testing_datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23_test/celeba_test/labels.csv', sep='\t', index_col=0)
        features, labels = lm.extract_features_labels()
        print("==============================Data loading complete==============================")
        return features, labels
    else:
        datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23/cartoon_set/labels.csv', sep='\t', index_col=0)
        return datasets

def solve_A1():
    print("==============================Task A1 start to solve==============================")
    features, labels = Load_data('A')
    
    # Supervised feature selection method:
    # Random forests
    return None

def solve_A2():
    
    return None

def solve_B1(datasets):
    
    return None
 
def solve_B2(datasets):
    
    return None

def main():
    # For part A
    solve_A1()
    solve_A2()
    
    # For part B
    datasets = Load_data('B')
    solve_B1(datasets)
    solve_B2(datasets)
    
if __name__ == "__main__":
    main()