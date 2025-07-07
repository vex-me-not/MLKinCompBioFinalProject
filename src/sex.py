import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import gc
import shap
import os
import joblib
import statsmodels.api as sm
import optuna
import lightgbm as lgb


from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

from tqdm.notebook import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from optuna.pruners import MedianPruner
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import r_regression
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder,RobustScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_predict
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score,f1_score, precision_score, recall_score, confusion_matrix, average_precision_score,accuracy_score
from scipy import stats


warnings.filterwarnings("ignore")

def download_data(brain_parts,directories_to_use,metadata_to_use):
    download_base = Path('../data/abc_atlas')
    abc_cache = AbcProjectCache.from_cache_dir(download_base)
    
    print('Current manifest is ',abc_cache.current_manifest)
    print('All available manifests: ',abc_cache.list_manifest_file_names)

    abc_cache.load_manifest('releases/20230630/manifest.json')
    print("We will be using manifest :", abc_cache.current_manifest)
    print("All available data directories:",abc_cache.list_directories)
    print("Directories to be used:", directories_to_use)

    for directory in directories_to_use:
        print(f"All available data files of {directory} directory: {abc_cache.list_data_files(directory)}")

    print("Metadata to be used:",abc_cache.list_metadata_files(metadata_to_use))
    
    print("Downloading the metadata\n")
    abc_cache.get_directory_metadata(metadata_to_use)
    
    for directory in directories_to_use:
        for brain_part in brain_parts:
            print(f"Dowloading the {directory} data file for brain part : {brain_part}")

            fname= directory + '-' + brain_part + '/log2'
            abc_cache.get_data_path(directory=directory, file_name=fname)


def find_outliers(series):
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    
    IQR = Q3 - Q1
    
    return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))



class IO:
    """
    This class implements the saving and the loading of the models. It needs a models directory in the constructor to work
    So, each instance of IO class coresponds to one specific directory.
    If you want to load or to save from another directory, you need to create a new instance
    """
    def __init__(self,models_dir):
        if models_dir is None:
            raise ValueError("Directory of models must be given!")

        self.models_dir=Path(models_dir)
        os.makedirs(self.models_dir,exist_ok=True)

    def __gen_filename(self,name,suffix):
        parts=[name]
        if suffix:
            parts.append(suffix)

        return self.models_dir/f"{'_'.join(parts)}.pkl"
    
    # method that saves a model adding a suffix suf to its name
    def save(self,model,name,suf=''):
        joblib.dump(model,self.__gen_filename(name,suffix=suf))

    # method
    def load(self,name,suf=''):
        return joblib.load(self.__gen_filename(name,suffix=suf))

class EarlyStopping:
    """
    This class implements the early stopping functionality in the optimize function of the optuna studies. The idea is that after patience
    trials have passed when no improvement has occured, the study will be stopped.
    """
    # the constructor of the class 
    def __init__(self, patience):
        self.patience = patience
        self.best = None
        self.no_impr_count = 0

    # the main mathod that gives this class the desired functionality. The trial parameter is needed because of the callbacks parameter in create study.
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if (self.best is None) or (study.best_value > self.best):
            self.best = study.best_value
            self.no_impr_count=0
        else:
            self.no_impr_count += 1

        if (self.no_impr_count >= self.patience):
            print(f"EARLY STOPPING: No improvement after {self.patience} trials!")
            study.stop()
            print(f"--> Best trial is {study.best_trial.number} with value: {study.best_trial.value} and parameters: {study.best_trial.params}\n")

class rnCV():

    """
    This class implements the repeated nested cross validation pipeline. The default values of r,n,k are those of the exercise but all can be
    changed through the constructor. The run_rnCV method is used to run the pipeline on the give data_df dataset, whilst the result summary is
    used to get the results of the pipeline in a dataframe.
    """
    # the constructor of the class. Notice that we don't have to give x and y, just the dataset
    def __init__(self,data_df,estimators,params,r=4,n=5,k=3,random_state=42):
        self.R=r
        self.N=n
        self.K=k
        self.x, self.y=keep_features(data_df=data_df,target='sex',to_drop='gene_identifier')
        self.estimators=estimators
        self.params=params
        self.random_state=random_state
        self.results={estim:[] for estim in estimators.keys()}

    # method used to calculate the various metrics required.
    def _compute_metrics(self,y_true,y_pred,y_prob):
        tn, fp, fn, tp=confusion_matrix(y_true, y_pred).ravel()
        specificity=tn / (tn + fp)
        npv=tn / (tn + fn)

        metrics={
            'MCC': matthews_corrcoef(y_true, y_pred),
            'ROC_AUC': roc_auc_score(y_true, y_prob),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Specificity': specificity,
            'NPV': npv,
            'PR_AUC': average_precision_score(y_true, y_prob)
        }

        return metrics
    
    def results_summary(self):
        summary = {}
        
        for model_name, all_metrics in self.results.items():
            df=pd.DataFrame(all_metrics)
            summary[model_name]=df.median().to_dict()
    
        summary_t=pd.DataFrame(summary).T
        
        return summary_t

    # the inner loop
    def _tune_model(self,x_train,y_train,model_name):
        # per optuna's documentation, we create the objective function that will be maximized
        def objective(trial):
            estimator=self.estimators[model_name]
            params=self.params[model_name](trial)
            model=estimator(**params)

            # we create k-folds whilst keeping the imbalances with StratifiedKFold
            strat_kfold=StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.random_state)
            scores=[]

            for fold_idx, (k_train_idx, k_val_idx) in enumerate(strat_kfold.split(x_train, y_train)):
                x_k_train, x_val = x_train.iloc[k_train_idx], x_train.iloc[k_val_idx]
                y_k_train, y_val = y_train.iloc[k_train_idx], y_train.iloc[k_val_idx]
                
                # we fit and we predict
                model.fit(x_k_train, y_k_train)
                preds=model.predict(x_val)

                # intermediate results, used in pruning
                score=balanced_accuracy_score(y_val, preds) 
                scores.append(score)

                # report intermediate result for pruning
                trial.report(score, step=fold_idx)
                
                # check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)        
        

        study=optuna.create_study(direction='maximize',study_name=model_name)
        study.optimize(objective, n_trials=30,timeout=180.0,callbacks=[EarlyStopping(patience=5)])
        
        return study.best_params        

    def run_rnCV(self):
        np.random.seed(self.random_state)
        
        # we run for all the repetitions
        for r in range(self.R):
            print(f"------ Repetition {r+1}/{self.R} ------\n")

            state=self.random_state + r
            strat_kfold=StratifiedKFold(n_splits=self.N, shuffle=True, random_state=state)

            # we perform the outer loop for each model
            for model_name in self.estimators.keys():
                # the outer loop
                for n_train_idx, n_test_idx in strat_kfold.split(self.x, self.y):
                    x_n_train, x_n_test=self.x.iloc[n_train_idx], self.x.iloc[n_test_idx]
                    y_n_train, y_n_test=self.y.iloc[n_train_idx], self.y.iloc[n_test_idx]

                    # we run the inner loops
                    best_params=self._tune_model(x_train=x_n_train,y_train= y_n_train,model_name=model_name)

                    model=self.estimators[model_name](**best_params)
                    
                    # we fit and we predict
                    model.fit(x_n_train, y_n_train)

                    preds=model.predict(x_n_test)
                    pred_probs=model.predict_proba(x_n_test)[:, 1]

                    # we compute the required metrics
                    metrics=self._compute_metrics(y_true=y_n_test,y_pred=preds,y_prob=pred_probs)
                    self.results[model_name].append(metrics)
                    

        return self.results

    # method used to find the optimal set of hyperparameters for the winner model
    def tune_winner(self,winner):

        # per optuna's documentation
        def objective(trial):
            estimator=self.estimators[winner]
            params=self.params[winner](trial)
            model=estimator(**params)

            strat_kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores=[]

            x_tune=self.x
            y_tune=self.y
            
            for fold_idx, (tune_train_idx, tune_val_idx) in enumerate(strat_kfold.split(x_tune, y_tune)):
                x_tune_train, x_tune_val= x_tune.iloc[tune_train_idx], x_tune.iloc[tune_val_idx]
                y_tune_train, y_tune_val= y_tune.iloc[tune_train_idx], y_tune.iloc[tune_val_idx]

                # we fit and we predict
                model.fit(x_tune_train, y_tune_train)
                preds=model.predict(x_tune_val)

                # intermediate results, used in pruning                
                score=balanced_accuracy_score(y_tune_val, preds)
                scores.append(score)

                # report intermediate result for pruning
                trial.report(score, step=fold_idx)
                    
                # check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)        
            

        study=optuna.create_study(direction='maximize',study_name="Winner:"+winner)
        study.optimize(objective, n_trials=60,timeout=600.0,callbacks=[EarlyStopping(patience=10)])
            
        return study.best_params


# method used to run the repeated nested cross validation pipeline
def perform_rnCV(path,res_dest='../data/rncv_SEX_summary_results.csv'):
    # we load the given dataset into a data frame
    df=pd.read_csv(path)

    # we define the estimators to be used
    estimators = {
        'LogisticRegression': LogisticRegression,
        'LDA': LinearDiscriminantAnalysis,
        'RandomForest': RandomForestClassifier,
        'LightGBM': lgb.LGBMClassifier
    }

    # we define the hyperparameter spaces, same as the ones we used in rnCV
    param_spaces = {
        'LogisticRegression': lambda trial: {
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['saga']),
            'C': trial.suggest_float('C', 1e-3, 1e0, log=True),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1)
        },
        'LDA': lambda trial: {'solver':trial.suggest_categorical('solver', ['svd']),
                              'tol':trial.suggest_float('tol', 5*1e-2, 1e-1, log=True)},
        'RandomForest': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        },
        'LightGBM': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'verbosity': trial.suggest_categorical('verbosity', [-1]),
            'is_unbalance': trial.suggest_categorical('is_unbalance',[True])
        }
    }

    # we initialize a rnCV class instance and we run_rnCV
    rncv=rnCV(data_df=df, estimators=estimators,params=param_spaces, r=2, n=4, k=3, random_state=42)
    results=rncv.run_rnCV()

    # we summarize and save the results in a dataframe
    summary=rncv.results_summary()
    summary.to_csv(res_dest)

    print("Summary:\n", summary)

# method used to tune the final winner model that we got through rnCV
def winner_tuning(df:pd.DataFrame,winner):
    # we define our estimators, same as the ones we used in rnCV
    estimators = {
        'LogisticRegression': LogisticRegression,
        'LDA': LinearDiscriminantAnalysis,
        'RandomForest': RandomForestClassifier,
        'LightGBM': lgb.LGBMClassifier
    }

    # we define the hyperparameter spaces, same as the ones we used in rnCV
    param_spaces = {
        'LogisticRegression': lambda trial: {
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['saga']),
            'C': trial.suggest_float('C', 1e-3, 1e0, log=True),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1)
        },
        'LDA': lambda trial: {'solver':trial.suggest_categorical('solver', ['svd']),
                              'tol':trial.suggest_float('tol', 5*1e-2, 1e-1, log=True)},
        'RandomForest': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        },
        'LightGBM': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'learning_rate': trial.suggest_float('learning_rate', 5*1e-3, 5*1e-1, log=True),
            'verbosity': trial.suggest_categorical('verbosity', [-1]),
            'is_unbalance': trial.suggest_categorical('is_unbalance',[True])
        }
    }

    # we initialize a rnCV class instance
    rncv=rnCV(data_df=df, estimators=estimators,params=param_spaces, r=10, n=5, k=3, random_state=42)
    winner_estim=estimators[winner] # we get the actual estimator, not just his name
    winner_params=rncv.tune_winner(winner=winner) # and we use rnCV tune_winner to tune it

    print(f'For model {winner} the best parameters are {winner_params}') # we print the best parameters for clarity

    return winner_estim(**winner_params)

def encode(data_df: pd.DataFrame,target='sex'):
    """
    Method used to encode the entries of the column 'diagnosis'
    Male --> 1
    Female --> 0
    """
    df=data_df
    
    # we use the LabelEncoder to encode the two classes
    df[target]=LabelEncoder().fit_transform(df[target]) # M->1, F->0

    return df
    

# method used to get the target of each dataset. Raises a ValueError if given target does not exist
def get_Y(data_df: pd.DataFrame,target='sex'):
    if target not in data_df.columns:
        raise ValueError("Please give a valid target!")
    
    return pd.DataFrame(data_df[target])

# method used to keep the features and the target. Return x and y
def keep_features(data_df: pd.DataFrame,target='sex',to_drop='gene_identifier'):
    tdrp=[]
    
    # we update our columns to be dropped with the list given
    if to_drop is not None:
        tdrp = [col for col in to_drop if col in data_df.columns]
    
    tdrp.append(target) # we add the target to the columns to be dropped
    Y=get_Y(data_df=data_df,target=target)
    x=data_df.drop(tdrp,axis=1)

    return x,Y

# method used to count how many times item appears in item_list
def count_apps(item,item_list:list):
    
    count=0
    
    for it in item_list:
        if (it==item):
            count += 1
    
    return count

# method used to count the wins of a method in a summary dataframe
def count_wins(summary:pd.DataFrame):
    # we count how many models we have
    total_models=summary.shape[0]

    dict_app={}

    # for each model, we count how many times it appears in the idxmax list
    for model_num in range(total_models):
        dict_app[model_num]=count_apps(model_num,summary.idxmax())
    
    return dict_app

# we transform the dictionairy we get from count_wins, to dictionairy with {model_name: number_of_wins} entries
def winner_dict(summary:pd.DataFrame):
    
    model_names=summary['Model']
    sum_no_name=summary.drop(columns='Model')

    win_dict=count_wins(summary=sum_no_name)

    for i in range(len(model_names)):
        win_dict[model_names[i]]=win_dict.pop(i)

    return win_dict

# method used to get the winner method, along with all the wins it scored
def get_winner(summary:pd.DataFrame):
    win_dict=winner_dict(summary=summary) # we get the dictionairy with all the wins

    # we sorted in descending order, based on the values it stores
    sorted_win_dict=dict(sorted(win_dict.items(), key=lambda item: item[1],reverse=True))
    winner=list(sorted_win_dict.keys())[0] # the key of the first entry is our winner

    return (winner,sorted_win_dict[winner])

# method used to replace/rename a specific column in a dataframe
def replace_column(df:pd.DataFrame,to_be_replaced,to_be_added):
    data_df=df
    
    # we ensure that the column to be replaced is part of the dataframe and that the new column does not alreadt exist
    if to_be_replaced in data_df.columns and (to_be_added not in data_df.columns):
        data_df.rename(columns={to_be_replaced:to_be_added},inplace=True)
    
    return data_df  


# method used to print the confidence interval of bootstrapping as actual intervals of a single metric
def metric_ci(y_true, y_pred, metric, is_proba=False, proba=None, n_samples=5000, seed=42):
    
    rng=np.random.RandomState(seed)
    stats=[]
    
    # we perform bootstrapping
    for sample in range(n_samples):
        
        idx=rng.choice(len(y_true), len(y_true), replace=True) # we choose a random index
        
        y_sample=y_true[idx] if isinstance(y_true, np.ndarray) else y_true.iloc[idx] # we check whether it's and np ndarray or a dataframe/series
        pred_sample=proba[idx] if is_proba else y_pred[idx] # we get the proper prediction type based on the metric
        
        stats.append(metric(y_sample, pred_sample)) # apply the metric given
    
    return np.percentile(stats, [2.5, 97.5]) # we return the two ends of the interval


# method used to print the confidence interval of bootstrapping as actual intervals([start,end]) of all metrics
def bootstrap_model_intervals(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):
    # our metrices
    metrics = {
        'Balanced Accuracy': (balanced_accuracy_score, False),
        'F1 Score': (f1_score, False),
        'Precision': (precision_score, False),
        'Recall': (recall_score, False),
        'MCC': (matthews_corrcoef, False),
        'ROC AUC': (roc_auc_score, True),
        'PR AUC': (average_precision_score, True)
    }

    # we get the x and y of the dev set
    x_dev, y_dev=keep_features(data_df=df_dev,target='sex',to_drop='gene_identifier')

    # we get the x and y of the val set    
    x_val, y_val=keep_features(data_df=df_val,target='sex',to_drop='gene_identifier')

    # we fit the model on dev
    model.fit(x_dev,y_dev)

    # predict on evaluation set
    y_pred=model.predict(x_val)
    y_proba=model.predict_proba(x_val)[:, 1]


    print("Bootstrapped 95% CIs (Model trained on dev, tested on val):")
    
    # we iterate over all the metric methods
    for name, (fn, is_proba) in metrics.items():
        # we calculate the interval of each specific metric using the actual function (fn) of the metric
        ci=metric_ci(
            y_val,
            y_pred if not is_proba else y_proba,
            fn,
            is_proba=is_proba,
            proba=y_proba if is_proba else None
        )
        # we print the method and the interval
        print(f"{name:15s}: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # specificity and NPV as point estimates (not bootstrapped here)
    tn, fp, fn_, tp=confusion_matrix(y_val, y_pred).ravel()
    
    specificity=tn / (tn + fp) if (tn + fp) > 0 else 0
    npv=tn / (tn + fn_) if (tn + fn_) > 0 else 0
    
    print(f"Specificity         : {specificity:.4f}")
    print(f"NPV                 : {npv:.4f}")


# method used to calculate the metrics score of all methods need for the violinplot
def metric_ci_plot(y_true, y_pred, proba, n_samples=1000, seed=42):
    rng = np.random.RandomState(seed)
    # the scores of each metric
    metrics = {
        'Balanced Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'MCC': [],
        'ROC AUC': [],
        'PR AUC': []
    }

    for sample in range(n_samples):
        
        idx=rng.choice(len(y_true), len(y_true), replace=True) # we choose a random index
        
        y_bs=y_true[idx] if isinstance(y_true, np.ndarray) else y_true.iloc[idx] # we check whether we have a ndarray or a Dataframe/Series
        y_pred_bs=y_pred[idx] # actual class prediction
        y_proba_bs=proba[idx] # class probabilities prediction

        # we calculate each metric by using either the actual class or the class probabilites prediction
        metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_bs, y_pred_bs))
        metrics['F1 Score'].append(f1_score(y_bs, y_pred_bs))
        metrics['Precision'].append(precision_score(y_bs, y_pred_bs))
        metrics['Recall'].append(recall_score(y_bs, y_pred_bs))
        metrics['MCC'].append(matthews_corrcoef(y_bs, y_pred_bs))
        metrics['ROC AUC'].append(roc_auc_score(y_bs, y_proba_bs))
        metrics['PR AUC'].append(average_precision_score(y_bs, y_proba_bs))

    return metrics

# method used to plot the confidence intervals of bootstrapping. It used violin plots
def bootstrap_model_plot(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):

    # we get the x and y of the dev set
    x_dev, y_dev=keep_features(data_df=df_dev,target='sex',to_drop='gene_identifier')

    # we get the x and y of the val set    
    x_val, y_val=keep_features(data_df=df_val,target='sex',to_drop='gene_identifier')

    # we fit the model on dev dataset
    model.fit(x_dev,y_dev)

    # we predict on evaluation set
    y_pred=model.predict(x_val) # actual classes prediction
    y_proba=model.predict_proba(x_val)[:, 1] # classe probabilities prediction

    # we run bootstrapping
    bootstrap_scores=metric_ci_plot(y_val, y_pred, y_proba, n_samples=5000)

    # we create a dataframe with the results to visualize with violin plots
    df_bootstrap = pd.DataFrame({
        'Metric': sum([[metric]*len(scores) for metric, scores in bootstrap_scores.items()], []),
        'Score': sum(bootstrap_scores.values(), [])
    })

    # we plot using violin plots
    plt.figure(figsize=(10, 6))
    
    sns.violinplot(data=df_bootstrap, x='Metric', y='Score', inner='quartile', palette='pastel')
    
    plt.title("Bootstrapped 95% CI Distributions (val set)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# method used to perform bootstrapping. It prints intervals and visualizes them with violin plots
def bootstrap_model(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):
    bootstrap_model_intervals(df_dev=df_dev,df_val=df_val,model=model)
    bootstrap_model_plot(df_dev=df_dev,df_val=df_val,model=model)

# method used to save the winner model. It actually saves a pipeline
def save_winner(train_path,test_path,winner,winner_name,saved_name='SEX_winner'):
    # the directory where the pipeline will be saved
    models_dir="../models"
    model_io=IO(models_dir) # class used to handle the actual saving

    df_train=pd.read_csv(train_path)
    df_test=pd.read_csv(test_path)

    common_columns = df_train.columns.intersection(df_test.columns)

    df_train=df_train[common_columns].copy()
    print(df_train.shape)
    df_test=df_test[common_columns].copy()
    print(df_test.shape)

    # we get the x and y of the given dataset
    x_full,y_full=keep_features(data_df=df_train)

    del df_train,df_test,common_columns
    # we drop the target from the features

    # we save the features we used so that the pipeline may use them during inference
    feature_columns=x_full.columns.tolist()
    joblib.dump(feature_columns, "../models/SEX_feature_columns.pkl")

    # the complete pipeline that will be saved. It includes column selection, preprocessing and the model
    winner_pipeline=Pipeline([
        ('model',winner)
    ])

    # we fit the model on the dataset
    winner.fit(X=x_full,y=y_full)
    
    print(f"Saving winner model ({winner_name}) with name {saved_name}.pkl")

    # we save the model
    model_io.save(model=winner_pipeline,name=saved_name)


# Method used to explain winner model that can be found in winner_src through SHAP values
def explain_winner(winner_src,dev_df,val_df,samples_explained,top_k,saved_name='SEX_winner'):
    # the directory where the winner lies
    models_dir=winner_src
    model_io=IO(models_dir) # this class will handle the loading

    winner=model_io.load(name=saved_name) # we load the model

    data,y_train=keep_features(data_df=dev_df)
    data_valid,y_valid=keep_features(data_df=val_df)
    
    # explain samples_explained examples from the validation set
    # each row is an explanation for a sample, and the last column in the base rate of the model
    # the sum of each row is the margin (log odds) output of the model for that sample
    shap_values = shap.TreeExplainer((winner.named_steps['model']).booster_).shap_values(data_valid.iloc[:samples_explained,:])
    shap_values.shape
    
    # compute the global importance of each feature as the mean absolute value
    # of the feature's importance over all the samples
    global_importances = np.abs(shap_values).mean(0)[:-1]

    # make a bar chart that shows the global importance of the top 20 features
    inds = np.argsort(-global_importances)
    f = plt.figure(figsize=(5,10))
    y_pos = np.arange(top_k)
    inds2 = np.flip(inds[:top_k], 0) # top 20
    plt.barh(y_pos, global_importances[inds2], align='center', color="#1E88E5")
    plt.yticks(y_pos, fontsize=13)
    plt.gca().set_yticklabels(data.columns[inds2])
    plt.xlabel('mean abs. SHAP value (impact on model output)', fontsize=13)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.show()

    shap.summary_plot(shap_values, data_valid.iloc[:samples_explained,:])

    # Dependency plot for top 20
    for i in reversed(inds2):
        shap.dependence_plot(i, shap_values, data_valid.iloc[:samples_explained,:])

# Method used to isolate the sex info for a specific area and a specifica experiment from a metadata file found in meatadata_path
def get_sexes(area,experiment,metadata_path):
    
    metadata_df=pd.read_csv(metadata_path)
    # metadata_df['donor_sex'] = metadata_df['donor_sex'].map({0: 'Female', 1: 'Male'})#.values # Female -> 0, # Male -> 1
    metadata_df=metadata_df[metadata_df['anatomical_division_label']==area]
    metadata_df=metadata_df[metadata_df['dataset_label']==experiment]
    
    sexes=metadata_df['donor_sex']


    del metadata_df
    gc.collect()

    save_path= '../data/'+experiment+'_sexes_'+ area + '.csv'
    
    sexes.to_csv(save_path,index=False)

# Encode and save the sex info found in sexes_path for a specific area and experiment
def save_encoded_sex(area,experiment,sexes_path):
    sexes=pd.read_csv(sexes_path)

    sexes=encode(data_df=sexes,target='donor_sex') # M->1 , F->0

    save_path= "../data/" + experiment +'ENCODED_sexes_'+area+'.csv'

    sexes.to_csv(save_path,index=False)

# Augment the anndata file at adata_path with the encoded sex info at encoded_sex_path and save it at dest_path
def create_sexed_adata(adata_path,encoded_sex_path,dest_path):
    
    adata=anndata.read_h5ad(adata_path)
    encoded_sex=pd.read_csv(encoded_sex_path)

    assert len(encoded_sex) == adata.n_obs, "Length mismatch between CSV and AnnData object"
    
    adata.obs['sex'] = encoded_sex.values
    
    del encoded_sex
    gc.collect()

    adata.write(dest_path)


# Method used to create a df from an anndata
def produce_df(hy_path,th_path,verbose=False,test=False):
    hypothalamus=anndata.read_h5ad(hy_path)

    if verbose is True:
        print(hypothalamus)
        print("\n\n-----Obs header-----\n\n",hypothalamus.obs.head())
        print("\n\n-----Var header-----\n\n",hypothalamus.var.head())


    thalamus=anndata.read_h5ad(th_path)

    if verbose is True:
        print(thalamus)
        print("\n\n-----Obs header-----\n\n",thalamus.obs.head())
        print("\n\n-----Var header-----\n\n",thalamus.var.head())

    adata_combined = thalamus.concatenate(hypothalamus, batch_key='source', batch_categories=['thalamus', 'hypothalamus'])

    del thalamus,hypothalamus
    gc.collect()

    state=42
    if test is True:
        state=41
    # Set a seed for reproducibility
    np.random.seed(state)

    # Generate a shuffled index
    shuffled_idx = np.random.permutation(adata_combined.n_obs)

    # Reorder the AnnData object
    adata_combined = adata_combined[shuffled_idx].copy()

    # Keep top 2000 genes, some sort of dimensionality reduction
    sc.pp.highly_variable_genes(adata_combined, n_top_genes=2000)
    adata_combined = adata_combined[:, adata_combined.var['highly_variable']]

    X = adata_combined.X.toarray() if not isinstance(adata_combined.X, np.ndarray) else adata_combined.X
    # X = adata_combined.X
    y = adata_combined.obs['sex'].values

    print(adata_combined.var_names)

    X_df=pd.DataFrame(X,columns=adata_combined.var_names)
    X_df['sex']=y

    del adata_combined,X,y
    gc.collect()

    return X_df

# Create
def sex_specific_region(region_path,verbose=False,test=False):
    region=anndata.read_h5ad(region_path)

    if verbose is True:
        print(region)
        print("\n\n-----Obs header-----\n\n",region.obs.head())
        print("\n\n-----Var header-----\n\n",region.var.head())
    

    state=42
    if test is True:
        state=41
    # Set a seed for reproducibility
    np.random.seed(state)

    # Generate a shuffled index
    shuffled_idx = np.random.permutation(region.n_obs)

    # Reorder the AnnData object
    region = region[shuffled_idx].copy()

    # Keep top 2000 genes, some sort of dimensionality reduction
    sc.pp.highly_variable_genes(region, n_top_genes=2000)
    region = region[:, region.var['highly_variable']]

    X = region.X.toarray() if not isinstance(region.X, np.ndarray) else region.X
    y = region.obs['sex'].values

    print(region.var_names)

    X_df=pd.DataFrame(X,columns=region.var_names)
    X_df['sex']=y

    del region,X,y
    gc.collect()

    return X_df

