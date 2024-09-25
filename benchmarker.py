import os
import sys

sys.path.append(os.path.abspath(".."))
os.environ['QT_QPA_PLATFORM']='offscreen'
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib
matplotlib.use('agg')
import sklearn
from utilities.util import cv_loo_splits, cv_kfold_splits, split_and_preprocess_dataset, merge_datasets
from plot_synthetic_data import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from viz import plot_input_data

class benchmarker():
    def __init__(self, dataset_dict, y, args, perturbed_mets = None, seed=0, path='.'):
        self.dataset_dict=dataset_dict
        self.y = y
        self.args = args
        self.num_subjects = self.y.shape[0]
        self.perturbed_mets = perturbed_mets
        self.seed = seed
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'

    def get_cv_splits(self):
        # Get train-test data splits for cross-val
        if self.args.cv_type == 'None':
            self.train_splits = [np.arange(self.num_subjects)]
            self.test_splits = self.train_splits
        elif self.args.cv_type == 'loo':
            self.train_splits, self.test_splits = cv_loo_splits(np.zeros(self.num_subjects), self.y)
        else:
            self.train_splits, self.test_splits = cv_kfold_splits(np.zeros(self.num_subjects), self.y,
                num_splits=self.args.kfolds, seed=self.seed)


    def train_ensemble_models(self, ensemble_model='RF'):
        grid_dict = {
            'bootstrap':[True],
            'n_estimators':[50,100],
            'max_depth':[None],
            'max_features':[None,'sqrt'],
            'min_samples_split':[2,9],
            'min_samples_leaf':[1,5],
            'learning_rate':[1e-5,5e-5,0.0001,5e-4,0.001,5e-3,0.01,0.05,0.1,0.5,1,5,10]
        }
        self.get_cv_splits()
        scores = {'F1': [], 'AUC': [], 'Case Acc':[],'Ctrl Acc':[]}
        coef_dfs = []
        pred_dfs = []
        best_params = []
        for i, (train_ids, test_ids) in enumerate(zip(self.train_splits, self.test_splits)):
            train_dict, test_dict = split_and_preprocess_dataset(self.dataset_dict, train_ids, test_ids, preprocess=True,
                                                                 logdir=self.path)

            plot_input_data(train_dict, self.path)
            x_tr = []
            for k, v in train_dict.items():
                x_tr.append(v['X'])
            X_tr = pd.concat(x_tr,axis=1)

            x_ts = []
            for k, v in test_dict.items():
                x_ts.append(v['X'])
            X_ts = pd.concat(x_ts,axis=1)

            if ensemble_model=='RF':
                rf = RandomForestClassifier()
            elif ensemble_model=='AdaBoost':
                rf = AdaBoostClassifier()
            elif ensemble_model=='GradBoost':
                rf = GradientBoostingClassifier()

            input_dict = rf._parameter_constraints
            rm_keys = list(set(list(grid_dict.keys())) - set(list(input_dict.keys())))
            for key in rm_keys:
                grid_dict.pop(key)
            rf_random = sklearn.model_selection.GridSearchCV(estimator=rf, param_grid=grid_dict, scoring=self.args.scorer,
                                                             cv=int(np.min([self.args.inner_folds,self.y.iloc[train_ids].sum()])),
                                                           )


            rf_random.fit(X_tr, self.y.iloc[train_ids])
            best_model = rf_random.best_estimator_
            f1 = best_model.score(X_ts, self.y.iloc[test_ids])
            coef_df = pd.DataFrame(best_model.feature_importances_.squeeze(), index = X_tr.columns.values, columns = ['Fold ' + str(i)])
            pred_probs = best_model.predict_proba(X_ts)
            try:
                auc = sklearn.metrics.roc_auc_score(self.y.iloc[test_ids].values, pred_probs[:, 1] > 0.5)
            except:
                auc=0
            coef_dfs.append(coef_df)
            scores['F1'].append(f1)
            scores['AUC'].append(auc)
            # a, f = by_hand_calc(self.y.iloc[test_ids].values, pred_probs[:, 1], 0.5)
            if args.cv_type!='loo':
                ctrls, case = self.y.iloc[test_ids].values==0, self.y.iloc[test_ids].values!=0
                case_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[case], (pred_probs[:, 1][case] > 0.5).astype(int)), 3)
                ctrl_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[ctrls], (pred_probs[:, 1][ctrls] > 0.5).astype(int)), 3)
            else:
                case_acc,ctrl_acc = 0,0
            scores['Case Acc'].append(case_acc)
            scores['Ctrl Acc'].append(ctrl_acc)
            best_params.append(pd.DataFrame(rf_random.best_params_, index = ['Fold {0}'.format(i)]))
            pred_dfs.append(pd.DataFrame({'True': self.y.iloc[test_ids].values, 'Pred probs': pred_probs[:, 1],
                                          'Outer Fold': [i]*len(test_ids)}, index = test_ids))

            if args.cv_type!='loo':
                pd.DataFrame(rf_random.cv_results_).to_csv(self.path + '/cv_results_fold_{0}.csv'.format(i))


        pd.concat(best_params).to_csv(self.path +'/best_params.csv'.format(self.args.run_name))
        pd.DataFrame(scores).to_csv(self.path +'/scores.csv'.format(self.args.run_name))
        coef_df_full = pd.concat(coef_dfs, axis=1)
        if self.perturbed_mets is not None:
            pert_df = coef_df_full.index.isin(self.perturbed_mets)
            coef_df_full = pd.concat({'Perturbed': coef_df_full.loc[self.perturbed_mets],
                                      'Un-perturbed': coef_df_full.loc[~pert_df]}, names = ['Metabolite perturbed'])
        coef_df_full.to_csv(self.path+'/coefs.csv'.format(self.args.run_name))
        pd.concat(pred_dfs).to_csv(self.path +'/preds.csv'.format(self.args.run_name))


    def train_l1_model(self):
        self.get_cv_splits()
        scores = {'F1': [], 'AUC': [], 'Case Acc':[],'Ctrl Acc':[]}
        coef_dfs = []
        pred_dfs = []
        l1_lambda = []
        lambda_min = 0.001
        path_len = 100
        l_max=100
        if args.taxa_tr == 'standardize':
            print('OTUS standardized')
        elif args.taxa_tr=='clr':
            print('OTUS clr')
        elif args.taxa_tr=='sqrt':
            print('OTUS sqrt')
        else:
            print('No OTU transform')
        for i, (train_ids, test_ids) in enumerate(zip(self.train_splits, self.test_splits)):
            train_dict, test_dict = split_and_preprocess_dataset(self.dataset_dict, train_ids, test_ids,
                                                                 preprocess=True,
                                                                 standardize_from_training_data=True,
                                                                 clr_transform_otus=args.taxa_tr=='clr',
                                                                 sqrt_transform=args.taxa_tr=='sqrt',
                                                                 logdir=self.path)
            plot_input_data(train_dict, self.path)

            x_tr = []
            for k, v in train_dict.items():
                x_tr.append(v['X'])
            X_tr = pd.concat(x_tr, axis=1)

            x_ts = []
            for k, v in test_dict.items():
                x_ts.append(v['X'])
            X_ts = pd.concat(x_ts, axis=1)

            if (len(np.unique(self.y.iloc[train_ids]))==1 or len(np.unique(self.y.iloc[test_ids]))==1) and args.cv_type!='loo':
                continue
            bval = True
            lam = 0
            while (bval):
                lam += 0.1
                model2 = LogisticRegression(penalty='l1', class_weight='balanced', C=1 / lam, solver='liblinear')
                try:
                    model2.fit(X_tr, self.y.iloc[train_ids])
                except:
                    print('error')
                if np.sum(np.abs(model2.coef_)) < 1e-8:
                    l_max = lam + 1
                    bval = False
            print(l_max)
            l_path = np.logspace(np.log10(l_max * lambda_min), np.log10(l_max), path_len)
            # l_path = np.logspace(np.log10(lambda_min), np.log10(l_max), path_len)
            Cs = [1/l for l in l_path]
            if X_tr.shape[0]<150:
                solver='liblinear'
            else:
                solver='saga'
            model = sklearn.linear_model.LogisticRegressionCV(cv = int(np.min([self.args.inner_folds,
                                                                           self.y.iloc[train_ids].sum()])),
                                                              penalty = 'l1', scoring = self.args.scorer,
                                                              Cs = Cs, solver = solver, class_weight='balanced',
                                                              random_state=self.seed)
            model.fit(X_tr, self.y.iloc[train_ids])
            with open(self.path + '/reg_params.txt','w') as f:
                f.write(str(1/model.C_))
            score = model.score(X_ts, self.y.iloc[test_ids])
            coef_df = pd.DataFrame(model.coef_.squeeze(), index = X_tr.columns.values, columns = ['Fold ' + str(i)])
            pred_probs = model.predict_proba(X_ts)

            f1_score = sklearn.metrics.f1_score(self.y.iloc[test_ids].values, pred_probs[:, 1] > 0.5, average='weighted')
            try:
                auc = sklearn.metrics.roc_auc_score(self.y.iloc[test_ids].values, pred_probs[:, 1]>0.5)
            except:
                auc=0
            coef_dfs.append(coef_df)
            scores['F1'].append(f1_score)
            scores['AUC'].append(auc)
            if args.cv_type!='loo':
                ctrls, case = self.y.iloc[test_ids].values==0, self.y.iloc[test_ids].values!=0
                case_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[case], (pred_probs[:, 1][case] > 0.5).astype(int)), 3)
                ctrl_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[ctrls], (pred_probs[:, 1][ctrls] > 0.5).astype(int)), 3)
                # a, f = by_hand_calc(self.y.iloc[test_ids].values, pred_probs[:, 1], 0.5)
            else:
                case_acc = 0
                ctrl_acc = 0
            scores['Case Acc'].append(case_acc)
            scores['Ctrl Acc'].append(ctrl_acc)

            l1_lambda.append(model.C_)
            pred_dfs.append(pd.DataFrame({'True': self.y.iloc[test_ids].values, 'Pred probs': pred_probs[:, 1],
                                          'Outer Fold': [i]*len(test_ids)}, index =self.y.index.values[test_ids]))
            with open(self.path + '/test_dataset.pkl','wb') as f:
                pkl.dump(X_ts, f)
            with open(self.path + '/train_dataset.pkl','wb') as f:
                pkl.dump(X_tr, f)
            with open(self.path + '/y.pkl','wb') as f:
                pkl.dump(self.y, f)


        pd.DataFrame(scores).to_csv(self.path +'/scores.csv')
        coef_df_full = pd.concat(coef_dfs, axis=1)
        if self.perturbed_mets is not None:
            pert_df = coef_df_full.index.isin(self.perturbed_mets)
            coef_df_full = pd.concat({'Perturbed': coef_df_full.loc[self.perturbed_mets],
                                      'Un-perturbed': coef_df_full.loc[~pert_df]}, names = ['Metabolite perturbed'])
        coef_df_full.to_csv(self.path +'/coefs.csv')
        pd.concat(pred_dfs).to_csv(self.path +'/preds.csv')
        print(l1_lambda)

def process_benchmark_coefs(path, syn_data_seed = None, seed_vec = None):
    seed_res = {}
    row_names=None
    if seed_vec is not None:
        seed_vec_names = [f'seed_{s}' for s in seed_vec]
    else:
        seed_vec_names = None
    for seed_folder in os.listdir(path):
        if '.' in seed_folder:
            continue
        if seed_vec_names is not None and seed_folder not in seed_vec_names:
            continue
        try:
            res_df = pd.read_csv(path + '/' + seed_folder + '/coefs.csv', index_col = 0)
        except:
            continue
        # if row_names is None:
        row_names = res_df.index.values
        seed_res[seed_folder] = res_df.loc[row_names]
    all_coefs = pd.concat(seed_res.values(), axis=1, join='inner')
    all_coefs_mean = all_coefs.mean(axis=1)
    all_coefs_median = all_coefs.median(axis=1)
    all_coefs_std = all_coefs.std(axis=1)
    all_coefs_5p = all_coefs.quantile(0.05, axis=1)
    all_coefs_95p = all_coefs.quantile(0.95, axis=1)
    all_coefs_25p = all_coefs.quantile(0.25, axis=1)
    all_coefs_75p = all_coefs.quantile(0.75, axis=1)
    sort_ix = np.flip(np.argsort(np.abs(all_coefs_median)))

    res = pd.concat([all_coefs_median, all_coefs_5p, all_coefs_95p, all_coefs_25p, all_coefs_75p,all_coefs_mean, all_coefs_std], keys=['median', '5%', '95%', '25%', '75%','mean','stdev'], axis=1)


    key_func = lambda x: x.abs()
    res_sorted = res.sort_values(by=['median','mean'], ascending=False, key = key_func)
    if syn_data_seed is not None and syn_data_seed!='':
        res_sorted.to_csv(path + '/' + f'coef_res_{syn_data_seed}.csv')
    elif seed_vec is not None:
        res_sorted.to_csv(path + '/' + f'coef_res_{"".join([str(s) for s in seed_vec])}.csv')
    else:
        res_sorted.to_csv(path + '/' + f'coef_res.csv')

def process_benchmark_results(path, syn_data_seed = None):
    seed_res = {}
    for seed_folder in os.listdir(path):
        if '.' in seed_folder:
            continue
        try:
            res_df = pd.read_csv(path + '/' + seed_folder + '/preds.csv', index_col = 0)
        except:
            continue
        cv_f1 = np.round(f1_score(res_df['True'], (res_df['Pred probs'] > 0.5).astype(int)), 3)
        cv_auc = np.round(roc_auc_score(res_df['True'], res_df['Pred probs']), 3)
        cv_f1_weighted=np.round(f1_score(res_df['True'], (res_df['Pred probs'] > 0.5).astype(int), average='weighted'), 3)
        # a, f = by_hand_calc(res_df['True'], res_df['Pred probs'], 0.5)
        ctrls, case = res_df['True'].values==0, res_df['True'].values!=0
        # case_acc = np.round(accuracy_score(res_df['True'].values[case], (res_df['Pred probs'].values[case] > 0.5).astype(int)), 3)
        # ctrl_acc = np.round(accuracy_score(res_df['True'].values[ctrls], (res_df['Pred probs'].values[ctrls] > 0.5).astype(int)), 3)
        seed_res[seed_folder] = {'F1': cv_f1, 'F1_weighted':cv_f1_weighted, 'AUC': cv_auc}

    seed_res['Mean'] = {'F1': np.round(np.mean([seed_res[s]['F1'] for s in seed_res.keys()]), 3),
                        'F1_weighted':np.round(np.mean([seed_res[s]['F1_weighted'] for s in seed_res.keys()]), 3),
                        'AUC': np.round(np.mean([seed_res[s]['AUC'] for s in seed_res.keys()]), 3),
                        }
    seed_res['St dev'] = {'F1': np.round(np.std([seed_res[s]['F1'] for s in seed_res.keys()]), 3),
                          'F1_weighted': np.round(np.std([seed_res[s]['F1_weighted'] for s in seed_res.keys()]), 3),
                        'AUC': np.round(np.std([seed_res[s]['AUC'] for s in seed_res.keys()]), 3),
                          }
    seed_res['Median'] = {'F1': np.round(np.median([seed_res[s]['F1'] for s in seed_res.keys()]), 3),
                          'F1_weighted': np.round(np.median([seed_res[s]['F1_weighted'] for s in seed_res.keys()]), 3),
                        'AUC': np.round(np.median([seed_res[s]['AUC'] for s in seed_res.keys()]), 3),
                          }

    seed_res['25%'] = {'F1': np.round(np.percentile([seed_res[s]['F1'] for s in seed_res.keys()], 25), 3),
                       'F1_weighted': np.round(np.percentile([seed_res[s]['F1_weighted'] for s in seed_res.keys()], 25), 3),
                        'AUC': np.round(np.percentile([seed_res[s]['AUC'] for s in seed_res.keys()], 25), 3),
                          }

    seed_res['75%'] = {'F1': np.round(np.percentile([seed_res[s]['F1'] for s in seed_res.keys()], 75), 3),
                       'F1_weighted': np.round(np.percentile([seed_res[s]['F1_weighted'] for s in seed_res.keys()], 75), 3),
                        'AUC': np.round(np.percentile([seed_res[s]['AUC'] for s in seed_res.keys()], 75), 3),
                          }
    pd.DataFrame(seed_res).T.to_csv(path + '/' + f'res_{syn_data_seed}.csv')
    pd.DataFrame(seed_res).T.to_csv(path + '/' + f'res.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
    parser.add_argument('--run_name', metavar='DIR',
                        help='run_name',
                        default='')
    parser.add_argument('--met_data', metavar='DIR',
                        help='path to metabolite dataset',
                        default='/Users/jendawk/Dropbox (MIT)/microbes-metabolites/datasets/FRANZOSA/processed/franzosa_pubchem/mets.pkl')
    parser.add_argument('--taxa_data', metavar='DIR',
                        help='path to taxanomic dataset',
                        default='/Users/jendawk/Dropbox (MIT)/microbes-metabolites/datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl')
    parser.add_argument('--seed', type=int, default=[0,1,2,3],
                        help='Set random seed for reproducibility', nargs = '+')
    parser.add_argument('--cv_type', type=str, default='kfold',
                        choices=['loo', 'kfold', 'None'],
                        help='Choose cross val type')
    parser.add_argument("--scorer",type=str, default='f1',choices=['f1', 'roc_auc', 'accuracy'],
                        help='metric for choosing best hyperparameter(s) in inner CV folds')
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--model', type=str, default='LR', choices=['RF','LR','GradBoost','AdaBoost'])
    parser.add_argument('--dtype', type=str, default=['metabs'], nargs='+') # OPTIONS
    parser.add_argument('--log_dir',type=str, default='logs/')
    parser.add_argument('--taxa_tr', type=str,default='none', choices=['standardize','clr','none','sqrt']) # OPTIONS
    parser.add_argument('--no_filter', type=int, default=0)

    args = parser.parse_args()

    print('')
    print('START')
    for k, v in args.__dict__.items():
        print(k, v)
    if args.run_name is None:
        args.run_name=''

    if len(args.dtype)>1:
        args.run_name += '_'.join(args.dtype)
        args.run_name += '_' + args.met_data.split('/')[-2]
    else:
        if 'taxa' in args.dtype and args.taxa_data is not None:
            args.run_name += args.dtype[0]
            args.run_name += '_' + args.taxa_data.split('/')[-2]
        if 'metabs' in args.dtype and args.met_data is not None:
            args.run_name += args.dtype[0]
            args.run_name += '_' + args.met_data.split('/')[-2]
    if args.met_data is not None:
        tmp = args.met_data.split('/')[-1]
    else:
        tmp = args.taxa_data.split('/')[-1]
    if len(tmp.split('_'))>2 or (len(tmp.split('_'))>1 and 'xdl' not in tmp):
        args.run_name += '_' + tmp.split('_')[0]

    args.run_name = args.run_name + '_' + args.scorer
    args.run_name += '_' + args.cv_type
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.isdir(args.log_dir + '/' + args.model ):
        os.mkdir(args.log_dir + '/' + args.model)

    if isinstance(args.seed, int):
        seed_vec = [args.seed]
    else:
        seed_vec = args.seed.copy()
    data_path = args.run_name
    print(data_path)
    ds_str = ''
    for seed in seed_vec:
        args.run_name = data_path + '/seed_{0}'.format(seed)
        print(args.met_data)
        print(args.taxa_data)

        if not os.path.isdir(args.log_dir + '/' + args.model + '/' + data_path):
            os.mkdir(args.log_dir + '/' + args.model + '/' + data_path)

        seed_path = args.log_dir + '/' + args.model + '/{0}/'.format(args.run_name)
        print(args.run_name)
        if 'seed_{0}'.format(seed) in os.listdir(args.log_dir + '/' + args.model + '/{0}/'.format(data_path)):
            # print(args.model + '/{0}/'.format(data_path))
            print(f'Seed {seed} training finished')
            continue

        dataset_dict = {}
        if 'metabs' in args.dtype or 'both' in args.dtype:
            dataset_dict['metabs'] = pd.read_pickle(args.met_data)
            if args.no_filter:
                dataset_dict['metabs']['preprocessing']={}
            if 'distances' not in dataset_dict['metabs'].keys():
                dataset_dict['metabs']['distances'] = dataset_dict['metabs']['tree_distance']
            if not isinstance(dataset_dict['metabs']['distances'], pd.DataFrame) and \
                    dataset_dict['metabs']['distances'].shape[0] == dataset_dict['metabs']['X'].shape[1]:
                dataset_dict['metabs']['distances'] = pd.DataFrame(dataset_dict['metabs']['distances'],
                                                                       index=dataset_dict['metabs']['X'].columns.values,
                                                                       columns=dataset_dict['metabs'][
                                                                           'X'].columns.values)
            mets = dataset_dict['metabs']['distances'].columns.values
            print(mets)
            dataset_dict['metabs']['X'] = dataset_dict['metabs']['X'][mets]
            print(f'{dataset_dict["metabs"]["X"].shape[1]} metabolites in data')


        if 'taxa' in args.dtype or 'both' in args.dtype:
            dataset_dict['taxa'] = pd.read_pickle(args.taxa_data)
            if args.no_filter:
                dataset_dict['taxa']['preprocessing']={}
            if args.full == 0:
                otus = dataset_dict['taxa']['distances'].columns.values
                dataset_dict['taxa']['X'] = dataset_dict['taxa']['X'][otus]
            print(f'{dataset_dict["taxa"]["X"].shape[1]} taxa in data')

        dataset_dict, y = merge_datasets(dataset_dict)
        if 'metabs' in dataset_dict.keys():
            y = pd.Series(y, index=dataset_dict['metabs']['y'].index.values)
        else:
            y = pd.Series(y, index=dataset_dict['taxa']['X'].index.values)

        # labels=y
        print(args.run_name)
        model = benchmarker(dataset_dict, y, args, perturbed_mets=None, seed = seed, path=seed_path)

        if args.model == 'LR':
            model.train_l1_model()
        else:
            model.train_ensemble_models(ensemble_model=args.model)

    if isinstance(args.seed, list) and len(args.seed)>1:
        process_benchmark_results(args.log_dir + '/' + args.model + '/' + data_path, syn_data_seed = ds_str)
        process_benchmark_coefs(args.log_dir + '/' + args.model + '/' + data_path, syn_data_seed = ds_str)
