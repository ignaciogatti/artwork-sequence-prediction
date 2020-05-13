import tensorflow as tf
import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from Prediction_model_feature import Prediction_model_feature
from Prediction_model_feature_multivariate import Prediction_model_feature_multivariate
from Prediction_model_feature_embeddings import Prediction_model_feature_embeddings


BASE_PATH = '/root/work/artwork_sequence/train_test_configuration'


class Sequence_prediction_factory(ABC):
    
    @abstractmethod
    def get_model(self):
        pass
    
    
    def get_x_features(self):
        return self._X.shape[1]
    
    def set_index(self, i):
        self._index = i
        
    def set_window_size(self, window_size):
        self._window_size = window_size
    
    
    def get_trained_weights_path(self):
        trained_weights_path = {
            'weights_folder' : os.path.join(self._CONFIG_PATH, 'config_'+str(self._window_size)+'/trained_model_weights')
        }
        return trained_weights_path


    def get_untrained_weights_path(self):
        '''
            It is useful because we load the model once and then we train with different configurations
        '''
        museum_sequence_path = {
            'weights_folder' : os.path.join(self._CONFIG_PATH, 'config_'+str(self._window_size)+'/untrained_model_weights')
        }
        return museum_sequence_path
    
    

class Sequence_prediction_univariate(Sequence_prediction_factory):
    
    def __init__(self, X, split_time, train_batch_size, val_batch_size, shuffle_buffer_size, CONFIG_PATH):
        self._X = X
        self._split_time = split_time
        self._train_batch_size = train_batch_size 
        self._val_batch_size = val_batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._CONFIG_PATH = CONFIG_PATH
    
    
    def get_model(self):
        return Prediction_model_feature(
            X=self._X,
            split_time=self._split_time,
            train_batch_size=self._train_batch_size, 
            val_batch_size=self._val_batch_size, 
            window_size=self._window_size, 
            shuffle_buffer=self._shuffle_buffer_size,
            n_features=1)
    
    
    
    
    
class Sequence_prediction_multivariate(Sequence_prediction_factory):
    
    def __init__(self, X, split_time, train_batch_size, val_batch_size, shuffle_buffer_size, n_influence_features, CONFIG_PATH):
        self._X = X
        self._split_time = split_time
        self._train_batch_size = train_batch_size 
        self._val_batch_size = val_batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._n_influence_features=n_influence_features
        self._CONFIG_PATH = CONFIG_PATH
    
    
    def get_model(self):
        return Prediction_model_feature_multivariate(
            X=self._X,
            split_time=self._split_time,
            train_batch_size=self._train_batch_size, 
            val_batch_size=self._val_batch_size, 
            window_size=self._window_size, 
            shuffle_buffer=self._shuffle_buffer_size,
            n_influence_features=self._n_influence_features,
            n_features=1)
    

class Sequence_prediction_embeddings(Sequence_prediction_factory):
    
    def __init__(self, X, split_time, train_batch_size, val_batch_size, shuffle_buffer_size, X_embeddings, CONFIG_PATH):
        self._X = X
        self._split_time = split_time
        self._train_batch_size = train_batch_size 
        self._val_batch_size = val_batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._X_embeddings = X_embeddings
        self._CONFIG_PATH = CONFIG_PATH
    
    
    def get_model(self):
        return Prediction_model_feature_embeddings(
            X=self._X,
            split_time=self._split_time,
            train_batch_size=self._train_batch_size, 
            val_batch_size=self._val_batch_size, 
            window_size=self._window_size, 
            shuffle_buffer=self._shuffle_buffer_size,
            X_embeddings=self._X_embeddings,
            n_features=1)