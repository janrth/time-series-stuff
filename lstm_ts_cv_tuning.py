
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.layers.experimental import preprocessing
import optuna
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
import pickle
from sklearn.preprocessing import MinMaxScaler

import preprocess_sliding_window


class TSCrossValidationTuningLSTM(object):
    
    def __init__(self, validation_length, test_length, n_steps, n_features, target, actions, df):
        self.validation_length = validation_length
        self.test_length = test_length
        self.n_steps = n_steps
        self.n_features = n_features
        self.df = df
        self.target = target
        self.actions = actions
        self.window_split = preprocess_sliding_window.DataPreProcessing()
        
    def save_best_params(self, params, pklname):
        with open('experiments/results_thesis/' + pklname, 'wb') as f:
            pickle.dump(params, f)

    def load_best_params(self, pklname):
        with open('experiments/results_thesis/' + pklname, 'rb') as f:
            return pickle.load(f)
        
    def lstm_model_v2(self, learning_rate, n_units_l1, n_units_l2, dropout_rate, batch_size, clipnorm):
        model = Sequential()
        #model.add(preprocessing.Normalization(input_shape=(self.n_steps, self.n_features)))
        model.add(LSTM(n_units_l1, activation='relu', return_sequences=True, input_shape=(self.n_steps, self.n_features)))  # Return sequences for next LSTM layer
        model.add(Dropout(dropout_rate))
        model.add(LSTM(n_units_l2, activation='relu'))  # Second LSTM layer
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                clipnorm=clipnorm,
                loss="mean_squared_error",
                metrics=["mean_squared_error"])
        return model
    
    def lstm_model_v1(self, learning_rate, n_units_l1, dropout_rate, batch_size, clipnorm):
        model = Sequential()
        model.add(LSTM(n_units_l1, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(
                optimizer=Adam(learning_rate=learning_rate, clipnorm=clipnorm),
                loss="mean_squared_error",
                metrics=["mean_squared_error"])
        return model
    
    
    def objective(self, trial):
        all_time_index = self.df.yrmo.unique()
        train_time_window = all_time_index[:-self.test_length]
        train_start = train_time_window[:-self.validation_length]

        # Optuna suggests values for the hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
        n_units_l1 = trial.suggest_int('n_units_l1', 32, 768)
        #n_units_l2 = trial.suggest_int('n_units_l2', 32, 768)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        clipnorm = trial.suggest_float('clipnorm', 0.5, 1)

        scores = []

        # Perform time-series cross-validation
        for t in tqdm(range(0, self.validation_length), desc='Validation time-windows loop'):
            # Create your train/validation split
            scaler = MinMaxScaler()
            
            df_train = self.df[(self.df.yrmo < all_time_index[len(train_start)+t])]
            df_validation = self.df[(self.df.yrmo <= all_time_index[len(train_start)+t])&
                   (self.df.yrmo > all_time_index[(len(train_start)-self.n_steps)+t])]
            
            df_train_temp = df_train[['account_id']].copy()
            df_validation_temp = df_validation[['account_id']].copy()
            
            df_train_temp_scaled = scaler.fit_transform(df_train[[self.target]+self.actions])
            df_validation_temp_scaled = scaler.transform(df_validation[[self.target]+self.actions])
            
            df_train_temp_scaled = pd.DataFrame(df_train_temp_scaled, index=df_train_temp.index, columns=[self.target]+self.actions)
            df_validation_temp_scaled = pd.DataFrame(df_validation_temp_scaled, index=df_validation_temp.index, columns=[self.target]+self.actions)
            
            df_train_scaled = pd.concat([df_train_temp, df_train_temp_scaled], axis=1)
            df_validation_scaled = pd.concat([df_validation_temp, df_validation_temp_scaled], axis=1)
            
            X_train, y_train = self.window_split.split_sequence_for_id(df_train_scaled, self.actions, self.target, self.n_steps)
            X_val, y_val = self.window_split.split_sequence_for_id(df_validation_scaled, self.actions, self.target, self.n_steps)
            
            print(df_validation.yrmo.max())
            print(len(X_val), len(y_val))
            print(np.mean(y_train), np.mean(y_val))
            # Build and train your model
            model = self.lstm_model_v1(learning_rate, n_units_l1, 
                                       #n_units_l2, 
                                       dropout_rate, batch_size,
                                       clipnorm)
            model.fit(X_train, y_train, epochs=2, 
                      #verbose=0, 
                      batch_size=batch_size
                      #callbacks=[TFKerasPruningCallback(trial, 'val_loss')]
                     )

            # Validate the model
            predictions = model.predict(X_val)
            score = mean_squared_error(y_val, predictions)
            scores.append(score)

        # Compute the average score over all time periods
        average_score = np.mean(scores)
        return average_score  # Optuna aims to minimize this value
    
    
    def run_optuna(self, n_trials, plotname, plotname_errors, logname, model_name, pklname, visualize_history=True):
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
        self.save_best_params(best_params, pklname)
        return best_params
    
    
    def train_best_model(self, pklname, plotname, X_train, y_train, epochs=10, plot_history=True):
        plt.rcParams["figure.figsize"] = (12,4)
        best_params = self.load_best_params(pklname)
        # Use the best hyperparameters to create a new model
        best_model = self.lstm_model_v1(list(best_params.values())[0], 
                                    list(best_params.values())[1],
                                    list(best_params.values())[2], 
                                    list(best_params.values())[3],
                                    list(best_params.values())[4]
                                   )

        history = best_model.fit(X_train, y_train, epochs=epochs)

        if plot_history:
            # plot loss history:
            metric = "loss"
            plt.figure()
            plt.plot(history.history[metric], label='train')
            #plt.plot(history.history["val_" + metric], label='validation')
            plt.title("LSTM: " + metric)
            plt.ylabel(metric, fontsize="large")
            plt.xlabel("epoch", fontsize="large")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig('experiments/results_thesis/'+plotname+'.png', bbox_inches='tight')
            plt.show()
            plt.close()

        return best_model


        