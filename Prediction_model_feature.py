import tensorflow as tf
import numpy as np


class Windowed_Dataset:

    #TODO: split X
    def __init__(self, X, split_time, window_size, shuffle_buffer, train_batch_size, val_batch_size):
        self._split_time = split_time
        self._x_train = X[:self._split_time]
        self._x_valid = X[self._split_time:]
        self._window_size = window_size
        self._train_dataset = self._create_train_dataset(train_batch_size, shuffle_buffer)
        self._val_dataset = self._create_validation_dataset(val_batch_size)
    
    
    def _windowed_dataset(self, series, batch_size):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(self._window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self._window_size + 1))
        return ds


    #Define a train dataset
    def _create_train_dataset(self, batch_size, shuffle_buffer):
        ds = self._windowed_dataset(self._x_train, batch_size)
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[-1:]))
        return ds.batch(batch_size).prefetch(1)

    
    #Define a validation dataset
    def _create_validation_dataset(self, batch_size):
        ds = self._windowed_dataset(self._x_valid, batch_size)
        ds = ds.map(lambda w: (w[:-1], w[-1:]))
        return ds.batch(batch_size).prefetch(1)

    
    #Doesn't work well with this dataset
    def normalize_data(self, x):
        x_mean = x.mean()
        x_std = x.std()
        x_norm = (x-x_mean)/x_std
        return x_norm
    
    
    def get_train_dataset(self):
        return self._train_dataset
    
    
    def get_val_dataset(self):
        return self._val_dataset


class Prediction_model_feature:
    
    
    def __init__(self, X, split_time, window_size, train_batch_size, val_batch_size, shuffle_buffer, name):
        self._dataset = Windowed_Dataset(X, split_time, window_size, shuffle_buffer, train_batch_size, val_batch_size)
        self._window_size = window_size
        self._name = name
        self._model = None

    
    def define_model(self):
        
        #To reset any variable in Tensorflow
        tf.random.set_seed(51)
        np.random.seed(51)
    
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=16, kernel_size=5,
                                  strides=1, padding="causal",
                                  activation="relu",
                                  input_shape=[None, 1]),
            tf.keras.layers.LSTM(self._window_size, return_sequences=True),
            tf.keras.layers.LSTM(self._window_size, return_sequences=True),
            tf.keras.layers.Dense(15, activation="relu"),
            tf.keras.layers.Dense(5, activation="relu"),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 400)
        ],
        name=self._name.replace(' ', '_'))
        
        return self._model
    
    
    def train_model(self, lr, epochs):
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        self._model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        history = self._model.fit(self._dataset.get_train_dataset(),
                                 epochs=epochs,
                                 validation_data=self._dataset.get_val_dataset())
        return history
    
    def get_model(self):
        if self._model != None:
            return self._model
        else:
            return self.define_model()