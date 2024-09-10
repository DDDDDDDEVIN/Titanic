import pandas as pd
import numpy as np
import pickle 
import itertools
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error

class DevinMengTuner:
    CP_TUNED_COMBO_PATH = 'tuned_comb.json'
    CP_BEST_COMBO_PATH = 'best_combo.json'
    CP_BEST_METRICS_PATH = 'best_metrics.json'
    def __init__(self):
        '''initialise object'''
        self.model = None
        self.model_type = None
        self.tunable_parameters_dict = {}
        self.non_tunable_parameters_dict = {}
        self.tuner_type = 'Grid'
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None
        self.tuned_combination_num = 0
        self.total_combination_num = 0
        self.curr_metrics_dict = {}
        self.best_metrics_dict = {}
        self.curr_param_dict = {}
        self.best_param_dict = {}
        self.CP_tuned_combination_list = []
        
        print("DevinMengTuner initialised")


    def set_model(self, model, model_type):
        # check valid input
        if model_type != 'Regression' and model_type != 'Classification':
            raise ValueError("model_type must be Regression or Classification, please try again")
        
        self.model = model
        self.model_type = model_type


    def set_parameters(self, tunable_parameters_dict, non_tunable_parameters):
        # check valid input
        if not isinstance(tunable_parameters_dict, dict):
            raise TypeError("input tunable_parameters_dict must be with type Dictionary, please try again")
        if not isinstance(non_tunable_parameters, dict):
            raise TypeError("input non_tunable_parameters_dict must be with type Dictionary, please try again")
        
        self.tunable_parameters_dict = tunable_parameters_dict
        self.non_tunable_parameters_dict = non_tunable_parameters


    def set_data(self, train_X, train_Y, test_X, test_Y):
        if not isinstance(train_X, pd.DataFrame):
            raise TypeError("input train_X must be with type DataFrame, please try again")
        if not isinstance(train_X, (pd.DataFrame, pd.Series)):
            raise TypeError("input train_Y must be with type DataFrame or Series, please try again")
        if not isinstance(test_X, pd.DataFrame):
            raise TypeError("input test_X must be with type DataFrame, please try again")
        if not isinstance(test_Y, (pd.DataFrame, pd.Series)):
            raise TypeError("input test_X must be with type DataFrame or Series, please try again")
        
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

    def  set_tuner(self, tuner_type):
        if tuner_type != 'Grid' and tuner_type != 'Random' and tuner_type != 'Bayesian':
            raise ValueError("input tunner_type must be Grid or Random or Bayesian, please try agian")
        
        self.tuner_type = tuner_type



    def tune(self):
        # check all attributes needed is set
        if self.model is None:
           raise ValueError("model is not set, please set_model")
        if self.model_type is None:
            raise ValueError("model_type is not set, please set_model")
        if self.tunable_parameters_dict is None:
            raise ValueError("tunable_parameters_dict is not set, please set_parameters")
        if self.train_X is None or self.train_X.empty:
           raise ValueError("train_X is not set, please set_data")
        if self.train_Y is None or self.train_Y.empty:
            raise ValueError("train_Y is not set, please set_data")
        if self.test_X is None or self.test_X.empty:
            raise ValueError("test_X is not set, please set_data")
        if self.test_Y is None or self.test_Y.empty:
            raise ValueError("test_Y is not set, please set_data")
        
        # check if there is checkpoint exist
        try:
            with open(self.CP_TUNED_COMBO_PATH, 'r') as file:
                self.CP_tuned_combination_list = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.CP_tuned_combination_list = []

        try:
            with open(self.CP_BEST_COMBO_PATH, 'r') as file:
                self.best_param_dict = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        try:
            with open(self.CP_BEST_METRICS_PATH, 'r') as file:
                self.best_metrics_dict = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        if self.tuner_type == 'Grid':
            self._grid_tune()
        if self.tuner_type == 'Random':
            self._random_tune()
        if self.tuner_type == 'Bayesian':
            self._bayesian_tune()

    def _grid_tune(self):
        # set non tunable parameters
        if self.non_tunable_parameters_dict:
            self.model.set_params(**self.non_tunable_parameters_dict)
        # create a list for all possible parameters combinations
        param_list = list(itertools.product(*self.tunable_parameters_dict.values()))
        self.total_combination_num = len(param_list)
        # looping through all combinatioons
        for curr_param in param_list:

            # check if current param is in checkpoint list
            checkpoint_found = 0
            for checkpoint_param in self.CP_tuned_combination_list:
                if checkpoint_param == list(curr_param):
                    checkpoint_found = 1
            if checkpoint_found:
                break

            # create a dictionary for current parameter-value pairs
            index = 0
            for param in self.tunable_parameters_dict.keys():
                self.curr_param_dict[param] = curr_param[index]
                index += 1
            # set current tunable params
            self.model.set_params(**self.curr_param_dict)
            # fit model
            self.model.fit(self.train_X, self.train_Y)
            # calculate evaluation matrics
            pred_Y = self.model.predict(self.test_X)
            if self.model_type == 'Regression':
                self._regression_metrics(pred_Y)
            if self.model_type == 'Classification':
                self._classification_metrics(pred_Y)
            # count combinations tuned
            self.tuned_combination_num += 1
            # calculate tuning pregress
            self._tuning_progress()
            print('----------------------------')
            # save checkpoint
            self.CP_tuned_combination_list.append(curr_param)
            with open(self.CP_TUNED_COMBO_PATH, 'w') as file:
                json.dump(self.CP_tuned_combination_list, file)
        self._print_best_combination()
        


    def _regression_metrics(self, pred_Y):
        # calculate r2 and rmse
        curr_r2 = r2_score(y_true=self.test_Y, y_pred=pred_Y)
        curr_rmse = mean_squared_error(y_true = self.test_Y, y_pred=pred_Y, squared=False)
        # record metrics as a dictionary
        self.curr_metrics_dict['R-Squared'] = curr_r2
        self.curr_metrics_dict['Root Mean Squared Error'] = curr_rmse
        # check if at least one best metrics are recorded
        if not self.best_metrics_dict:
            self.best_metrics_dict = self.curr_metrics_dict
            self.best_param_dict = self.curr_param_dict
            #check if current params perfroms better than best params
        elif self.curr_metrics_dict['R-Squared'] > self.best_metrics_dict['R-Squared']:
            self.best_metrics_dict = self.curr_metrics_dict
            self.best_param_dict = self.curr_param_dict
        self._print_evaluation()


    def _classification_metrics(self, pred_Y):
        # calculate f1 and accuracy score
        curr_f1 = f1_score(y_true=self.test_Y, y_pred=pred_Y)
        curr_accuracy = accuracy_score(y_true=self.test_Y, y_pred=pred_Y)
        # record metrics as a dictionary
        self.curr_metrics_dict['Accuracy'] = curr_accuracy
        self.curr_metrics_dict['F1-Score'] = curr_f1
        # check if at least one best metrics are recorded
        if not self.best_metrics_dict:
            self.best_metrics_dict = self.curr_metrics_dict.copy()
            self.best_param_dict = self.curr_param_dict.copy()
        # check if current params performs better than best params
        elif curr_accuracy > self.best_metrics_dict['Accuracy']:
            self.best_metrics_dict = self.curr_metrics_dict.copy()
            self.best_param_dict = self.curr_param_dict.copy()
        self._print_evaluation()
        with open(self.CP_BEST_COMBO_PATH, 'w') as file:
            json.dump(self.best_param_dict, file)
        with open(self.CP_BEST_METRICS_PATH, 'w') as file:
            json.dump(self.best_metrics_dict, file)

    def _print_evaluation(self):
        # print curr combo and best combo
        print('Fit succeed.')
        print('Current Parameter Combination:')
        for key, value in self.curr_param_dict.items():
            print(f'{key}: {value}', end='; ')
        print('')
        print('Current Performance Metrics:')
        for key, value in self.curr_metrics_dict.items():
            print(f'{key}: {value}', end='; ')
        print('')
        print('Best Performance Metrics:')
        for key, value in self.best_metrics_dict.items():
            print(f'{key}: {value}', end ='; ')
        print('')

    def _tuning_progress(self):
        tuned_percentage = round((float(self.tuned_combination_num) / self.total_combination_num) * 100, ndigits= 2)
        print(f'Already tuned {self.tuned_combination_num} out of {self.total_combination_num} in total')
        print(f'Progress: {tuned_percentage}%')

    def _print_best_combination(self):
        print('Best Performance Metrics:')
        for key, value in self.best_metrics_dict.items():
            print(f'{key}: {value}', end ='; ')
        print('')
        print('Best Performance Parameter Combination:')
        for key, value in self.best_param_dict.items():
            print(f'{key}: {value}', end ='; ')
        print('')

    def _random_tune(self):
        print('not done yet')


    def _bayesian_tune(self):
        print('not done yet')