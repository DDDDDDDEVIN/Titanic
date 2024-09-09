import pandas as pd
import numpy as np
import pickle 

from sklearn.model_selection import train_test_split

class DevinMengTuner:

    def __init__(self):
        '''initialise object'''
        self.model = None
        self.model_type = None
        self.tunable_paramters_dict = {}
        self.non_tunable_parameters_dict = {}
        self.tuner_type = 'Grid'
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None
        print("DevinMengTuner initialised")


    def set_model(self, model, model_type):
        # check valid input
        if model_type != 'Regression' or model_type != 'Classification':
            raise ValueError("model_type must be Regression or Classification, please try again")
        
        self.model = model
        self.model_type = model_type


    def set_parameters(self, tunable_parameters_dict, non_tunable_parameters):
        # check valid input
        if tunable_parameters_dict is not dict:
            raise TypeError("input tunable_parameters_dict must be with type Dictionary, please try again")
        if non_tunable_parameters is not dict:
            raise TypeError("input non_tunable_parameters_dict must be with type Dictionary, please try again")
        
        self.tunable_parameters_dict = tunable_parameters_dict
        self.non_tunable_parameters_dict = non_tunable_parameters


    def set_data(self, train_X, train_Y, test_X, test_Y):
        if train_X is not pd.DataFrame:
            raise TypeError("input train_X must be with type DataFrame, please try again")
        if train_Y is not pd.DataFrame or pd.Series:
            raise TypeError("input train_Y must be with type DataFrame or Series, please try again")
        if test_X is not pd.DataFrame:
            raise TypeError("input test_X must be with type DataFrame, please try again")
        if test_Y is not pd.DataFrame or pd.Series:
            raise TypeError("input test_X must be with type DataFrame or Series, please try again")
        
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_X

    def  set_tuner(self, tuner_type):
        if tuner_type != 'Grid' and tuner_type != 'Random' and tuner_type != 'Bayesian':
            raise ValueError("input tunner_type must be Grid or Random or Bayesian, please try agian")
        
        self.tuner_type = tuner_type



    def tune(self):
        # check all attributes needed is set
        if not self.model:
            raise ValueError("model is not set, please set_model")
        if not self.model_type:
            raise ValueError("model_type is not set, please set_model")
        if not self.tunable_parameters_dict:
            raise ValueError("tunable_parameters_dict is not set, please set_parameters")
        if not self.train_X:
            raise ValueError("train_X is not set, please set_data")
        if not self.train_Y:
            raise ValueError("train_Y is not set, please set_data")
        if not self.test_X:
            raise ValueError("test_X is not set, please set_data")
        if not self.test_Y:
            raise ValueError("test_Y is not set, please set_data")

        if self.tuner_type == 'Grid':
            self._grid_tune(self)
        if self.tuner_type == 'Random':
            self._random_tune(self)
        if self.tuner_type == 'Bayesian':
            self._bayesian_tune(self)

    def _grid_tune(self):
        # set non tunable parameters
        if self.non_tunable_parameters_dict:
            self.model.set_param(**self.non_tunable_parameters_dict)


        for parameter, value_list in self.tunable_parameters_dict.items():
            for value in value_list:
                curr_param_dict = {}
                curr_param_dict[parameter] = value

        

