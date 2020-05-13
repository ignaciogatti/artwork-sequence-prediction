from Abstract_sequence_generation_rnn import Abstract_sequence_generator_rnn
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
import tensorflow as tf
from Prediction_model_feature_embeddings import Prediction_model_feature_embeddings
from IPython.display import clear_output
import numpy as np
import pandas as pd
import os


class Sequence_generator_rnn_embedding(Abstract_sequence_generator_rnn):
    
    def __init__(self, window_size, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time, X_embeddings, conv_filter=20, lstm_filter=40, dense_filter=20, prediction_length=1):
        super().__init__(window_size, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time, conv_filter, lstm_filter, dense_filter, prediction_length)
        self._X_embeddings = X_embeddings
        self.models = self._load_model()
    
    
    def _create_rnn_model(self):
        return Prediction_model_feature_embeddings(
                X=self._X,
                split_time=self._split_time,
                train_batch_size=self._batch_size, 
                val_batch_size=self._batch_size, 
                window_size=self._window_size, 
                shuffle_buffer=self._shuffle_buffer_size,
                X_embeddings=self._X_embeddings)
    
    
    def _define_x_features(self, feature):
        x_features = self._X_tour[:, feature]
        #Stack feature to predict with embedding
        x_features = np.hstack(tup=(x_features.reshape((-1,1)), self._X_embedding_tour))
        return x_features
        