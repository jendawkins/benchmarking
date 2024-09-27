## Package dependencies
- python 3.11
- ete3 (v3.1.2)
- ete_toolchain 
- matplotlib (v3.3.4)
- bitarray (v2.3.0)
- hmmer (v3.3.2)
- guppy3 (v3.1.0)
- rdkit
- statsmodels (v0.12.2)
- scipy (v1.5.4)
- scikit-learn (v0.24.2)
- R

## Overview
This repository is for learning a binary outcome from joint metabolomics and sequencing profiles via supervised machine learning. Specifically, given metabolomics and either 16s or metagenomics data for a set of subjects, **benchmarker.py** performs either lasso logistic regression, random forest, adaptive boosting, or gradient boosting on the multi-modal profiles using nested cross-validation to optimize hyperparameters. 

To use **benchmarker.py**, the user must first process data using **process_data.py**. **process_data.py** also can be used to obtain metabolomic and phylogenetic classifications that can be useful for interpretability of downstream analyses. 

## Process data from a config file
Create a config file for your dataset. **16s_and_metabolomics.cfg** and **metagenomics_and_metabolomics.cfg** are two example config files. Options for config files are further explained below.
After specifying config file, run the following command:
> python ./process_data.py --config_file <path/to/your_config_file.cfg>

This creates pickle files in output folder specified in the config file for the metabolite data and the sequence data. Note that if 'coviariate_variable' is specified in the config file, there will be a pickle file for each covariate that the dataset is split by. If not, there is just one file for the metabolites and one for the sequences.

Both pickle files are a dictionary with the following keys:
- **'X'**: a Pandas DataFrame with columns of the metabolite levels or sequence counts for each subject (subjects are rows)
  - If process_before_training is TRUE, X will be filtered and transformed according to the parameters set in **[metabolite_preprocessing]** or **[sequence_preprocessing]**
- **'y'**: a Pandas Series with binary class labels for each subject
  - You can specify which class (or classes) recieves the "1" label by the **outcome_positive_value** setting in **[meta_data]**
- **'taxonomy'**: a Pandas dataframe with columns corresponding to the columns of "X", and rows corresponding to taxonomy.
  - For taxa, this is the standard "Kingdom, phylum, ..., genus, species" taxonomy.
  - For metabolites, the taxonomy is created from "Classy-Fire" designations, which designate the following:
    - kingdom, superclass, class, subclass, level 5, level 6, level 7, level 9, level 9, level 10
- **'tree'**: the phylogenetic tree for taxa, or a tree based on classy-fire classifications (where branch lengths are set to 1). Note: if a phylogenetic tree for 16s data cannot be made (due to issues with pplacer) or a reference tree is not provided for WGS data, the tree for taxa will be based on taxonomy, with branch lengths set to 1.
- **'preprocessing'**: if "**process_before_training**" is TRUE, this is an empty dictionary. If not, this is a dictionary of the parameters set in the [metabolite_preprocessing] or [sequence_preprocessing] sections so that preprocessing can be done during model training
- **'class_labels'**: dictionary mapping binary label (1 or 0) to original class labels found in **[meta_data][subject_data]**

## Run benchmarking
To run machine learning, run **benchmarker.py** with the following options:
 - **met_data**: path to pickle file with processed metabolomics dataset
 - **taxa_data**: path to pickle file with processed 16s or WGS dataset
 - **seed**: random seed as integer, or list of random seeds to run several seeds at once
 - **cv_type**: "kfold" (default) for k-fold cross validation, or "loo" for leave-one-out cross-validation
 - **scorer**: Metric used to optimize hyperparameters. Choices are "f1" (default), "roc_auc", or "accuracy"
 - **kfolds**: number of k-folds to use (default is 5)
 - **model**: which ML model to run. Choices are: "LR", "RF", "GradBoost", and "AdaBoost"
 - **data_type**: "metabs" to run on only metabolomics data, "taxa" to run on only taxanomic data, or "both" (default) to run on joint data
 - **log_dir**: path to directory for storing results
 - **taxa_tr**: if running LR, which transformation to preform on the taxa relative abundances before standardization. Options are "clr" for centered log-ratio transform, "sqrt" (default) for square root transform, or "none" (not recommended)

 For example, to run 10 seeds of lasso logistic regression on joint metabolomic and metagenomic profiles from Franzosa et al. with set defaults, run: 
 > python3 ./benchmarking.py --met_dat ./datasets/FRANZOSA/processed/TEST/mets.pkl --taxa_data ./datasets/FRANZOSA/processed/TEST/taxa.pkl --data_type both --seed 0 1 2 3 4 5 6 7 8 9 --data_type both --model LR

## Config file options for processing data
> **[description]** \
> **tag**:*name of your dataset*\
> **in_path**:*(optional) path of input data folder*\
> **out_path**:*(optional) path to save processed data; defaults to current working directory if not specified*
>
> **[meta_data]**\
> **subject_data**:path/to/subject_meta-data_file, as .csv or .tsv. *Data should have samples as the rows and the outcome as a column. If data is to be split on a coviariate, the covariate should also be included as a column.*\
> **outcome_variable**:*Column name in subject_data that has subject outcomes*\
> **outcome_positive_value**:*(optional) If outcome values are not ints (0s and 1s), specify the outcome string that signifies the outcome has occured*\
> **outcome_negative_value**:*(optional) Can specify instead of outcome_positive_value or can specify both. Specifying both is useful if data has some samples with a third outcome; these samples won't be included in processed data*\
> **sample_id_column**:*(optional) Use to signify the sample IDs column in subject_data if the sample IDs are not the index of subject_data*\
> **covariate_variable**:*(optional) If needed, specify covariate variable that data should be split by (i.e. race for vaginal microbiome data)*\
> **covariate_positive_value**:*(optional) If specifying covariate_variable and covariate values are not ints (0s,1s) specify covariate positivd value*\
> **covariate_negative_value**:*(optional) Can specify instead of outcome_positive_value or can specify both (i.e. if some samples have third covariate that shouldn't be in processed data)*\
>
> **[sequence_data]**\
> **data_type**:*Specify if data is from 16s amplicon sequencing or WGS/metagenomics. Options are* [16s, WGS]\
> **data**:path/to/sequence_data_file as .csv or .tsv. *Format expected is rows of samples and columns of sequences. Sequences can
be named with identifiers or with the actual RNA sequence string.*\
> **taxonomy**:(optional) path/to/input_taxonomy_file as csv or tsv\
> **tree**:*(optional) newick_tree. If not specified, pplacer will run to create tree (this may take a bit of time)*\
> **reference_tree**: *(optional) newick tree. Phylogenetic reference tree containing most/all taxa in data*\
> **reference_mapper**: *(optional) .tsv or .csv. If node names in "reference tree" do not correspond to column names of data, provide mapper file for tree node name --> data column name*\
> **rna_sequence_strings**:path to file with RNA sequences*. *Required only if column labels in sequence_data are not the actual RNA sequence string. Should be either a .csv/.tsv file with ASV labels (matching column labels in data) as the index and sequences as the first column; or a .fa/.txt file with the format "ASV_label>\nSEQUENCE" for all sequences*\

> **[sequence_preprocessing]** *(optional section specifying data filtering/transformations)*\
> **process_before_training**:*(optional, options=*[True, False]*) whether to process data during training according to training set (thereby preventing data leakage), or beforehand. Note that prediction results may be inflated if set to True. Defaults to False if not specified.*\
> **percent_present_in**:*(optional) percent of samples the microbe must be present above limit_of_detection; below this, microbes are filtered out*\
> **limit_of_detection**:*(optional) defaults to 0 if not provided*\
> **cov_percentile**:*(optional) if specified, get each metabolite's coefficient of variation and keep only metabolites in the top cov_percentile*
> **transformations**:*(optional, options=*[relative_abundance, None]*). Whether to ransform sequence data to relative abundance*\
>
> **[metabolite_data]**\
> **data**:${description:in_path}/*metabolite data file, as tsv or csv. Samples should be as rows and metabolites as columns.*\
> **meta_data**:(optional) path to meta-data file for metabolites, as tsv or csv. 
> Indices should correspond to metabolite feature names in metabolitee data. 
> Data frame columns should include either HMBD ids (labeled 'HMDB' or 'hmdb'), 
> KEGG ids (labeled 'KEGG' or 'kegg') or inchikeys (labeled 'InChIKey' or 'inchikey').
> If not specified, cannot calculate taxonomy or metabolite tree.\
> **taxonomy**:*(optional) Path to classy fire classifications for this set of metabolites. If not available, data processor will get these from the metabolite meta data*\
>
> **[metabolite_preprocessing]** *(Optional section specifying metabolite preprocessing)*\
> **process_before_training**:*(optional, options=*[True, False]*) whether to process data during training according to training set (thereby preventing data leakage), or beforehand. Defaults to False if not specified*\
> **percent_present_in**:*(optional) percent of samples the metabolite must be present above "limit_of_detection" to not be filtered out*\
> **limit_of_detection**:*(optional) defaults to 0 if not specified*\
> **cov_percentile**:*(optional) if specified, get each metabolite's coefficient of variation and keep only metabolites in the top cov_percentile*\
> **transformations**:*(optional, options=*[log, standardize] *or both (i.e. log, standardize)*

