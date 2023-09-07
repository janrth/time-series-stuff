import xgboost as xgb

import optuna
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
import category_encoders as ce
import scienceplots
plt.style.use('science')

class TSCrossValidationTuningXGBOOST(object):
    
    def __init__(self, df, target, actions, validation_length, test_length, categorical=False):
        self.df = df
        self.target = target
        self.actions = actions
        self.validation_length = validation_length
        self.test_length = test_length
        self.categorical = categorical
        
    def save_best_params(self, params, pklname):
        with open('experiments/results_thesis/' + pklname, 'wb') as f:
            pickle.dump(params, f)

    def load_best_params(self, pklname):
        with open('experiments/results_thesis/' + pklname, 'rb') as f:
            return pickle.load(f)
        
    def objective(self, trial):
        all_time_index = self.df.yrmo.unique()
        train_time_window = all_time_index[:-self.test_length]
        train_start = train_time_window[:-self.validation_length]

        param = {
                "verbosity": 0,  # To silence XGBoost
                "objective": "reg:squarederror",  # For regression
                "eval_metric": "rmse",  # Evaluation metric
                "booster": trial.suggest_categorical("booster", ["gbtree"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", 1, 15),
                "n_estimators": trial.suggest_int("n_estimators", 50, 2000),  # Number of trees
                "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            }

        scores = []

        # Perform time-series cross-validation
        for t in tqdm(range(0, self.validation_length), desc='Validation time-windows loop'):
            # Create your train/validation split
            df_train = self.df[(self.df.yrmo < all_time_index[len(train_start)+t])].copy()
            df_validation = self.df[(self.df.yrmo == all_time_index[len(train_start)+t])].copy()
            
            X_train = df_train[self.actions]
            y_train = df_train[self.target]
            
            X_val = df_validation[self.actions]
            y_val = df_validation[self.target]
            
            if self.categorical:
                # create the encoder instance
                encoder = ce.CatBoostEncoder(cols=['speciality'])
                # fit and transform on training data
                X_train = encoder.fit_transform(X_train, y_train)
                X_val = encoder.transform(X_val, y_val)
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_val, label=y_val)
            
            print(df_validation.yrmo.max())
            print(len(X_val), len(y_val))
            print(np.mean(y_train), np.mean(y_val))
            # Build and train your model
            model = xgb.train(param, dtrain)

            # Validate the model
            predictions = model.predict(dvalid)
            score = mean_squared_error(y_val, predictions)
            scores.append(score)

        # Compute the average score over all time periods
        average_score = np.mean(scores)
        return average_score  # Optuna aims to minimize this value
    
    
    def run_optuna(self, n_trials, plotname, plotname_errors, logname, model_name, visualize_history=True):
        # Create a study
        plt.rcParams["figure.figsize"] = (12,4)
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

        # Optimize the study, i.e., find the best hyperparameters
        study.optimize(self.objective, n_trials=n_trials)

        # Print the result
        best_params = study.best_params
        print(best_params)

        if visualize_history:
            #plt.figure(figsize=(12, 7)) 
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title('Optimisation history: '+model_name)
            plt.tight_layout()
            plt.savefig('experiments/'+plotname+'.png', bbox_inches='tight')
            plt.show()
            
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title('Hyperparameter importances: '+model_name)
            plt.tight_layout()
            plt.savefig('experiments/results_thesis/'+plotname_errors+'.png', bbox_inches='tight')
            
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create a file handler
        handler = logging.FileHandler('experiments/'+logname+'.log')
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(handler)
        logging.info(best_params)
        logger.removeHandler(handler)
        return best_params
        
    def train_best_model(self, best_params, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.actions)
        model = xgb.train(best_params, dtrain)
        return model
        