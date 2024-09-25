import os
import pandas as pd
import copy
import pickle as pkl

for root,dirs,files in os.walk('../datasets/'):
    if 'map4' in root:
        d_paths = [os.path.join(root,p) for p in files if 'mets.pkl' in p]
        for path in d_paths:
            dataset=pd.read_pickle(path)
            dist = dataset['distances']
            print(f'Dataset at {path} has {dist.shape[0]} metabolites originally')
            mets_to_keep=[]
            for met in dist.index.values:
                row = dist.loc[met]
                if sum(row.isna().values)== len(row)-1:
                    continue
                else:
                    mets_to_keep.append(met)
            new_dataset = copy.deepcopy(dataset)
            new_dataset['distances'] = dist[mets_to_keep].loc[mets_to_keep]
            print(f'After removing NAs, dataset has {new_dataset["distances"].shape[0]} metabolites')
            print('')
            with open(path, 'wb') as f:
                pkl.dump(new_dataset,f)
