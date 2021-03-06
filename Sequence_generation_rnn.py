from Abstract_sequence_generation_rnn import Abstract_sequence_generator_rnn
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
import tensorflow as tf
from Prediction_model_feature import Prediction_model_feature
from IPython.display import clear_output
import numpy as np
import pandas as pd
import os


class Sequence_generator_rnn(Abstract_sequence_generator_rnn):
    
    def __init__(self, window_size, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time, conv_filter=20, lstm_filter=40, dense_filter=16, prediction_length=1):
        super().__init__(window_size, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time, conv_filter, lstm_filter, dense_filter, prediction_length)
        self.models = self._load_model()
    
    
    def _create_rnn_model(self):
        return Prediction_model_feature(
                X=self._X,
                split_time=self._split_time,
                train_batch_size=self._batch_size, 
                val_batch_size=self._batch_size, 
                window_size=self._window_size, 
                shuffle_buffer=self._shuffle_buffer_size)
    
    
    
    def _define_x_features(self, feature):
        return self._X_tour[:,feature]
        