import pandas as pd
import itertools
import numpy as np
import copy
from utilities.data_utils import filter_by_presence_func, filter_by_cov_func, transform_func, get_epsilon
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
import matplotlib.pyplot as plt
import os

def recurse_parents(p):
    if len(p.get_leaves())>=2:
        return p
    else:
        recurse_parents(p.up)


def CLR(X):
    # X can be matrix of counts or relative abundances
    # X should be a matrix with rows of samples and columns of OTUs/ASVs
    indices, columns = None, None
    if isinstance(X, pd.DataFrame):
        indices, columns = X.index.values, X.columns.values
        X =X.values

    if any(np.sum(X,1)>2):
        if not np.all(X):
            X = X+1.
        X = X/np.sum(X,1)
    else:
        if not np.all(X):
            eps = get_epsilon(X)
            X = X+eps

    D = X.shape[1]
    Id = np.eye(D)
    Jd = np.ones(D)
    X_clr = np.log(X)@(Id - (1/D)*Jd)
    if indices is not None:
        X_clr = pd.DataFrame(X_clr, index=indices, columns=columns)
    return X_clr



def split_and_preprocess_dataset(dataset_dict, train_ixs, test_ixs, preprocess=True, logdir=None,
                                 standardize_otus=False, standardize_from_training_data=True,
                                 clr_transform_otus=False, sqrt_transform=False):
    """
    splits dataset into train and test splits given and filters/transforms
    based on the training set IF that filtering was not done in data preprocessing.
    Inputs:
    - dataset_dict: output dataset dictionary from process_data.py
    - train_ixs: list of index locations for train set
    - test_ixs: list of index locations for test set

    Outputs:
    - train_dataset_dict: dict with all the same keys as dataset dict, but with just the training subjects and the
    features after filtering
    - test_dataset_dict: dictionary with test subjects and filtered/transformed based on test set

    """
    new_dict = {}
    test_dict = {}
    write_lines=[]
    for key, dataset in dataset_dict.items():
        write_lines.append(key + '\n')
        new_dict[key] = copy.deepcopy(dataset)
        test_dict[key] = copy.deepcopy(dataset)
        if isinstance(dataset['X'], pd.DataFrame):
            new_dict[key]['X'] = dataset['X'].iloc[train_ixs,:]
            test_dict[key]['X'] = dataset['X'].iloc[test_ixs, :]
        else:
            new_dict[key]['X'] = dataset['X'][train_ixs,:]
            test_dict[key]['X'] = dataset['X'][test_ixs, :]
        if isinstance(dataset['y'], pd.Series):
            new_dict[key]['y'] = dataset['y'].iloc[train_ixs]
            test_dict[key]['y'] = dataset['y'].iloc[test_ixs]
        else:
            new_dict[key]['y'] = dataset['y'][train_ixs]
            test_dict[key]['y'] = dataset['y'][test_ixs]
        if 'X_mask' in dataset.keys():
            new_dict[key]['X_mask'] = dataset['X_mask'][train_ixs,:]
            test_dict[key]['X_mask'] = dataset['X_mask'][test_ixs,:]

        if 'preprocessing' in dataset.keys() and dataset['preprocessing'] and preprocess:
            print(key)

            ls = f'{dataset["y"].sum()} subjects have label=1; {len(dataset["y"])-dataset["y"].sum()} subjects have label=0'
            write_lines.append(ls + '\n')
            temp_tr, temp_ts = new_dict[key]['X'], test_dict[key]['X']
            if isinstance(dataset['preprocessing'], dict) and 'percent_present_in' in dataset['preprocessing'].keys():
                ppi=dataset['preprocessing']['percent_present_in']
                if isinstance(dataset['preprocessing'], dict) and 'lod' in dataset['preprocessing'].keys():
                    lod=dataset['preprocessing']['lod']
                else:
                    lod=0
                # lod=0
                temp_tr, temp_ts = filter_by_presence_func(temp_tr, temp_ts, ppi, lod)
                ls = 'After filtering {3} to keep only {3} with levels > {0} in {1}\% of participants, {2} remain'.format(lod, ppi, temp_tr.shape[1], key)
                write_lines.append(ls + '\n')
                print(ls)
            # print(temp_tr.shape)
            if isinstance(dataset['preprocessing'], dict) and 'cov_percentile' in dataset['preprocessing'].keys():
                cp=dataset['preprocessing']['cov_percentile']
                if cp>0:
                    temp_tr, temp_ts = filter_by_cov_func(temp_tr, temp_ts, cp)
                    ls = 'After filtering {2} to keep only {2} with coefficients of variation in the top {0} percentile of participants, {1} {2} remain'.format(
                        cp, temp_tr.shape[1], key)
                    write_lines.append(ls + '\n')
                    print('After filtering {2} to keep only {2} with coefficients of variation in the '
                          'top {0} percentile of participants, {1} {2} remain'.format(
                        cp, temp_tr.shape[1], key))
            # print(temp_tr.shape)
            if key == 'metabs':
                temp_tr, temp_ts = transform_func(temp_tr, temp_ts, standardize_from_training_data = standardize_from_training_data, log_transform=True)

            elif key == 'otus' or key=='taxa':
                # if (temp_tr>1).any().any():
                #     print('Transforming to RA')
                temp_tr = temp_tr.divide(temp_tr.sum(1),axis='index')
                temp_ts = temp_ts.divide(temp_ts.sum(1),axis='index')
                if clr_transform_otus:
                    temp_tr = CLR(temp_tr)
                    temp_ts = CLR(temp_ts)
                    temp_tr, temp_ts = transform_func(temp_tr, temp_ts,
                                                      standardize_from_training_data=standardize_from_training_data,
                                                      log_transform=False)
                    print("WARNING: CLR OTUS")
                elif sqrt_transform:
                    temp_tr = np.sqrt(temp_tr)
                    temp_ts = np.sqrt(temp_ts)
                    temp_tr, temp_ts = transform_func(temp_tr, temp_ts,
                                                      standardize_from_training_data=standardize_from_training_data,
                                                      log_transform=False)
                    print("WARNING: SQRT OTUS")
                    # temp_tr, temp_ts = temp_tr.T, temp_ts.T
                elif standardize_otus:
                    temp_tr, temp_ts = transform_func(temp_tr, temp_ts, standardize_from_training_data = standardize_from_training_data, log_transform=True)
                    print("WARNING: STANDARDIZED OTUS")

            new_dict[key]['X'], test_dict[key]['X'] = temp_tr, temp_ts
            kept_features = new_dict[key]['X'].columns.values
            if 'distances' in new_dict[key].keys():
                kept_sm = [k for k in kept_features if k in new_dict[key]['distances'].columns.values]
                new_dict[key]['distances'] = new_dict[key]['distances'][kept_sm].loc[kept_sm]
                test_dict[key]['distances'] = test_dict[key]['distances'][kept_sm].loc[kept_sm]

            new_dict[key]['variable_names'] = kept_features
            test_dict[key]['variable_names'] = kept_features

            ls = f'{temp_tr.shape[1]} features in {key} dataset after filtering. Test set has {temp_ts.shape[0]} samples and train set has {temp_tr.shape[0]} samples'
            write_lines.append(ls)
            print(f'{temp_tr.shape[1]} features in {key} dataset after filtering. Test set has {temp_ts.shape[0]} samples and train set has {temp_tr.shape[0]} samples')
            write_lines.append('\n')

    # if 'otus' in new_dict.keys():
    #     new_dict['otus']['X'] = np.divide(new_dict['otus']['X'].T, new_dict['otus']['X'].sum(1)).T
    #     test_dict['otus']['X'] = np.divide(test_dict['otus']['X'].T, test_dict['otus']['X'].sum(1)).T

    if logdir is not None:
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        with open(logdir + '/data_processing.txt','w') as f:
            f.writelines(write_lines)
    return new_dict, test_dict


def merge_datasets(dataset_dict):
    """
    Ensures that both datasets have the same samples in the data X and labels y

    Inputs:
    - dataset_dict: dictionary of dataset-dictionaries generated from process_data.py, where each key is the data type (i.e. {'metabs': metabolite-data-dictionar, 'otus': sequence-data-dictionary)

    Outputs:
    - dataset_dict: same dictionary, but with ensuring that indices in X and y for both datasets match
    - y: [N_subjects] outcome labels (i.e. 'y' in either dataset dictionary)

    """
    yls_all = []
    yls = []
    for key, dataset in dataset_dict.items():
        yls.append(set(dataset['X'].index.values))
        yls_all.extend(list(dataset['X'].index.values))
    yls_all = np.unique(yls_all)

    yixs_tmp = list(set.intersection(*yls))
    yixs = [y for y in yls_all if y in yixs_tmp]
    for key, dataset in dataset_dict.items():
        dataset_dict[key]['y']=dataset_dict[key]['y'][~dataset_dict[key]['y'].index.duplicated(keep='first')]
        dataset_dict[key]['X'] = dataset_dict[key]['X'][~dataset_dict[key]['X'].index.duplicated(keep='first')]
        dataset_dict[key]['y'] = dataset['y'].loc[yixs]
        dataset_dict[key]['X'] = dataset['X'].loc[yixs]

    y = dataset_dict[key]['y']
    return dataset_dict, y

# Get stratified kfold train/test splits for cross-val
def cv_kfold_splits(X, y, num_splits=5, seed=42):
    """
    Inputs:
    - X: [N_subjects x N_features] data array, OR [N_subjects] array of zeros
    - y: [N_subjects] array of outcomes
    - num_splits: number of k-folds (defaults to 5)
    - seed: random seed

    Outputs:
    - train_ids: list of k lists, where each list is the train index location for that fold
    - test_ids: list of k lists, where each list is the test index location for that fold

    """
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

    train_ids = list()
    test_ids = list()
    for train_index, test_index in skf.split(X, y):
        train_ids.append(train_index)
        test_ids.append(test_index)

    return train_ids, test_ids


# Get leave-one-out train/test splits for cross-val
def cv_loo_splits(X, y):
    """
    Inputs:
    - X: [N_subjects x N_features] data array, OR [N_subjects] array of zeros
    - y: [N_subjects] array of outcomes

    Outputs:
    - train_ids: list of k lists, where each list is the train index location for that fold
    - test_ids: list of k lists, where each list is the test index location for that fold

    """
    skf = LeaveOneOut()

    train_ids = list()
    test_ids = list()
    for train_index, test_index in skf.split(X, y):
        train_ids.append(train_index)
        test_ids.append(test_index)

    return train_ids, test_ids


def plot_input_data(dataset_dict, outpath):
    for name, dataset in dataset_dict.items():
        fig, ax = plt.subplots()
        X, y = dataset['X'], dataset['y']
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        ax.hist(X[y == 0].flatten(), alpha=0.5, label='-', bins=20)
        ax.hist(X[y == 1].flatten(), alpha=0.5, label='+', bins=20)
        ax.legend()
        fig.savefig(outpath+ '/' + name + '_training_data.png')