from Abstract_sequence_generation_rnn import Abstract_sequence_generator_rnn
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
import tensorflow as tf
from Prediction_model_feature_multivariate import Prediction_model_feature_multivariate
from IPython.display import clear_output
import numpy as np
import pandas as pd
import os


class Sequence_generator_rnn_multivariate(Abstract_sequence_generator_rnn):
    
    def __init__(self, window_size, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time, n_influence_features, conv_filter=40, lstm_filter=64, dense_filter=20):
        
        super().__init__(window_size, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time, conv_filter, lstm_filter, dense_filter)
        self._n_influence_features = n_influence_features
        self.models = self._load_model()

    
    def _create_rnn_model(self, i):
        return Prediction_model_feature_multivariate( 
            X=self._X, split_time=self._split_time, 
            train_batch_size=self._batch_size, 
            val_batch_size=self._batch_size, 
            window_size=self._window_size, 
            shuffle_buffer=self._shuffle_buffer_size, 
            name="feature " + str(i), 
            index=i, 
            n_influence_features=self._n_influence_features, 
            n_features=1)
            
    
    
    def _predict_features(self):
        
        predicted_features = []
        for feature in range(self._n_features):
            #Define feature to take into account for prediction
            x_influence_features = self._models[feature].get_indexes_features()
            x_influence_features = np.insert(arr=x_influence_features, obj=0, values=feature)
            
            #Predict feature i
            x_feature = self._X_tour[:,x_influence_features]
            rnn_forecast = self._model_forecast(self._models[feature].get_model(), x_feature, self._window_size, self._batch_size)
            rnn_forecast = rnn_forecast[1:,-1]
            
            predicted_features.append(rnn_forecast)
        
        self._forecast_matrix = np.stack(predicted_features)
        self._forecast_matrix = self._forecast_matrix.T
        return self._forecast_matrix
        