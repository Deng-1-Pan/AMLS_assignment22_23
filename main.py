import pandas as pd

import src.landmarks as lm

def Load_data(type):
    if type == 'A':
        features, labels = lm.extract_features_labels()
        return features, labels
    else:
        datasets = pd.read_csv('./Datasets/dataset_AMLS_22-23/cartoon_set/labels.csv', sep='\t', index_col=0)
        return datasets

def solve_A1(datasets):
    
    return

def solve_A2(datasets):
    
    return

def solve_B1(datasets):
    
    return

def solve_B2(datasets):
    
    return

def main():
    # For part A
    features, labels = Load_data('A')
    solve_A1(features, labels)
    solve_A2(features, labels)
    
    # For part B
    datasets = Load_data('B')
    solve_B1(datasets)
    solve_B2(datasets)
    
if __name__ == "__main__":
    main()