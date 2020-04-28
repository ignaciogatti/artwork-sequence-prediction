from Prediction_model_feature import Prediction_model_feature
import tensorflow as tf
import numpy as np
import os
import pickle

class Prediction_model_feature_embeddings(Prediction_model_feature):
    
    def __init__(self, X, split_time, window_size, train_batch_size, val_batch_size, shuffle_buffer, X_embeddings, n_features=1):
        
        super().__init__(X, split_time, window_size, train_batch_size, val_batch_size, shuffle_buffer)
        self._X_embeddings = X_embeddings
        self._n_features += self._X_embeddings.shape[1]
        
    
    def _define_x_features(self):
        self._x_features = self._X[:, self._index]
        #Stack feature to predict with embedding
        self._x_features = np.hstack(tup=(self._x_features.reshape((-1,1)), self._X_embeddings))
        return self._x_features