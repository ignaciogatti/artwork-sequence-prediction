from Prediction_model_feature import Prediction_model_feature
import tensorflow as tf
import numpy as np
import os
import pickle

class Prediction_artist_model(Prediction_model_feature):
    
    def _define_x_features(self):
        self._x_features = self._X

        return self._x_features
    
    
    def define_model(self, conv_filter=20, lstm_filter=40, dense_filter=20, prediction_length=1):

        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)
        self._prediction_length = prediction_length

        self._model = tf.keras.models.Sequential([
          tf.keras.layers.Conv1D(filters=conv_filter, kernel_size=5,
                              strides=1, padding="causal",
                              activation="relu",
                              input_shape=[self._window_size, self._n_features]),
          tf.keras.layers.LSTM(lstm_filter, return_sequences=True),
          tf.keras.layers.LSTM(lstm_filter//2),
          tf.keras.layers.Dense(dense_filter, activation="relu"),
          tf.keras.layers.Dense(8, activation="relu"),
          tf.keras.layers.Dense(prediction_length, name="prediction"),
          tf.keras.layers.Lambda(lambda x: x * 400)
        ],
        name="sequence_attist")

        return self._model