import tensorflow as tf
import numpy as np

class Prediction_model_feature:
    
    
    def __init__(self, x_train, window_size, batch_size, shuffle_buffer_size):
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.x_train = x_train
        #Doesn't work well with this dataset
        self.x_train_norm = self.normalize_data()
        self.train_set = self.windowed_dataset(self.x_train, self.window_size, self.batch_size, self.shuffle_buffer_size)
    
    
    def normalize_data(self):
        x_train_mean = self.x_train.mean()
        x_train_std = self.x_train.std()
        x_train_norm = (self.x_train-x_train_mean)/x_train_std
        return x_train_norm
    
    
    def windowed_dataset(self, series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[-1:]))
        return ds.batch(batch_size).prefetch(1)
    
    
    def define_model(self):
        
        #To reset any variable in Tensorflow
        tf.random.set_seed(51)
        np.random.seed(51)
    
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                                  strides=1, padding="causal",
                                  activation="relu",
                                  input_shape=[None, 1]),
            tf.keras.layers.LSTM(self.window_size, return_sequences=True),
            tf.keras.layers.LSTM(self.window_size, return_sequences=True),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 400)
        ])
        
        return self.model
    
    
    def train_model(self, lr, epochs):
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        self.model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        history = self.model.fit(self.train_set,epochs=epochs)
        return history