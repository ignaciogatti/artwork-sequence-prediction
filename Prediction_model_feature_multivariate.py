from Prediction_model_feature import Prediction_model_feature
import tensorflow as tf
import numpy as np
import os
import pickle

class Prediction_model_feature_multivariate(Prediction_model_feature):
    
    def __init__(self, X, split_time, window_size, train_batch_size, val_batch_size, shuffle_buffer, name, index, n_influence_features, n_features=1):
        
        super().__init__(X, split_time, window_size, train_batch_size, val_batch_size, shuffle_buffer, name, index)
        self._n_features += n_influence_features
        self._n_influence_features = n_influence_features
        
    
    def _define_x_features(self):
        self._x_features = self._X[:, self._index]
        #Define influence features
        self._indexes_features = np.random.choice(np.delete(np.arange(self._X.shape[1]), self._index), self._n_influence_features, replace=False)
        x_inflence_features = self._X[:, list(self._indexes_features)]
        #Stack feature to predict with influences features
        self._x_features = np.hstack(tup=(self._x_features.reshape((-1,1)), x_inflence_features))
        return self._x_features
    
    
    def save_weights(self, museum_sequence_path):
        super().save_weights(museum_sequence_path)
        #Save influence features index for the feature 
        x_influence_feature_folder = os.path.join(museum_sequence_path['weights_folder'], 'x_influence_feature_'+str(self._index))
        with open(x_influence_feature_folder, 'wb') as fp:
            pickle.dump(self._indexes_features, fp)

    
    def load_weights(self, museum_sequence_path):
        super().load_weights(museum_sequence_path)
        #Load influence features index for the feature 
        x_influence_feature_folder = os.path.join(museum_sequence_path['weights_folder'], 'x_influence_feature_'+str(self._index))
        #Load weights
        with open (x_influence_feature_folder, 'rb') as fp:
            self._indexes_features = pickle.load(fp)
        return self._model