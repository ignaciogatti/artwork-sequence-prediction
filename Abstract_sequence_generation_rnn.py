from Sequence_generator_models import Sequence_generator_class
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
import tensorflow as tf
from Prediction_model_feature import Prediction_model_feature
from IPython.display import clear_output
import numpy as np
import pandas as pd
import os
from abc import abstractmethod


class Abstract_sequence_generator_rnn(Sequence_generator_class):
    
    def __init__(self, window_size, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time, conv_filter=16, lstm_filter=32, dense_filter=16, prediction_length=1):
        self._name= "Sequence_generator_rnn"
        self._window_size = window_size
        self._df_all_metadata = df_all_metadata
        self._all_data_matrix = all_data_matrix
        self._museum_sequence_path = museum_sequence_path
        
        self._split_time = split_time
        self._X = X
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._conv_filter = conv_filter
        self._lstm_filter = lstm_filter
        self._dense_filter = dense_filter
        self._prediction_length = prediction_length
        

    @abstractmethod    
    def _create_rnn_model(self):
        pass
    
    
    
    def _load_model(self):
        self._n_features = self._X.shape[1]
        #Create model
        self._model = self._create_rnn_model()
        self._model.define_model(
            conv_filter=self._conv_filter, 
            lstm_filter=self._lstm_filter, 
            dense_filter=self._dense_filter, 
            prediction_length=self._prediction_length
            )
        return self._model

    
    
    def _model_forecast(self, model, series, window_size, batch_size):
        if len(series.shape) == 1:
            series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(self._window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self._window_size))
        ds = ds.map(lambda w: (w[:]))
        ds = ds.batch(batch_size)
        forecast = model.predict(ds)
        return forecast
  
    
    def _drop_selected_artwork(self, indexes, df_all_metadata, all_data_matrix):
    
        #Remove from metadata
        df_removed = df_all_metadata.copy()
        df_removed = df_removed.drop(indexes)
        df_removed = df_removed.reset_index(drop=True)

        #Remove ftom code matrix
        code_matrix = all_data_matrix.copy()
        code_matrix = np.delete(code_matrix, indexes, 0)

        return df_removed, code_matrix
    
    
    @abstractmethod
    def _define_x_features(self, feature):
        pass

    
    def _predict_features(self):
        
        predicted_features = []
        for feature in range(self._n_features):
            #Load weights for feature i
            self._model.set_index(feature)
            self._model.load_weights(self._museum_sequence_path)

            #Define feature to take into account for prediction
            x_feature = self._define_x_features(feature)
    
            #Predict feature i
            rnn_forecast = self._model_forecast(self._model.get_model(), x_feature, self._window_size, self._batch_size)
            #rnn_forecast = rnn_forecast.reshape((-1))
            rnn_forecast = rnn_forecast[1:,-1]
            
            predicted_features.append(rnn_forecast)
        
        self._forecast_matrix = np.stack(predicted_features)
        self._forecast_matrix = self._forecast_matrix.T
        return self._forecast_matrix
        
    
    def predict_tour(self):
        
        
        #Dataframe with the tour
        self._df_predicted_tour = pd.DataFrame({ 'title' : [],
                         'author' : [],
                         'sim_value' : [],
                         'tour_path': [],
                         'image_url':[]})
       
        ##List with the artworks's code that belongs to the tour
        self._predicted_code_list =[]

        
        #Check if window size is bigger than the tour length
        if (self._X_tour.shape[0] - self._window_size < 1 ):
            return self._df_predicted_tour
                
        #Made a copy of the data to keep the data safe
        df_all_metadata = self._df_all_metadata.copy()
        all_data_matrix = self._all_data_matrix.copy()
        
        
        #Predict features
        self._forecast_matrix = self._predict_features()

        for i in range(self._forecast_matrix.shape[0]):
            #Find code
            code = self._forecast_matrix[i].reshape((1,-1))

            #Compute cosine similarity
            sim_matrix = cosine_similarity(code, all_data_matrix)

            #sort indexes
            sort_index = np.argsort(sim_matrix.reshape((-1,)))

            #Find most similar
            sim_artwork_index = sort_index[-1]

            #Save in dataframe 
            self._df_predicted_tour = self._df_predicted_tour.append(
                {'title' : df_all_metadata.iloc[sim_artwork_index]['title'],
                 'author': df_all_metadata.iloc[sim_artwork_index]['author'],
                 'tour_path':df_all_metadata.iloc[sim_artwork_index]['tour_path'],
                 'image_url':df_all_metadata.iloc[sim_artwork_index]['image_url'],
                 'sim_value':sim_matrix[:,sim_artwork_index][0]
                }, 
               ignore_index=True)

            #Save predicted artwork's code
            self._predicted_code_list.append(all_data_matrix[sim_artwork_index])

            #Remove selected artworks
            df_all_metadata, all_data_matrix = self._drop_selected_artwork([sim_artwork_index], df_all_metadata, all_data_matrix)


        return self._df_predicted_tour
    
    
    def get_predicted_tour_matrix(self):
        #No tour predicted because the window size was too big
        if len(self._predicted_code_list) == 0:
            return np.array([])
        
        forecast_matrix = np.stack(self._predicted_code_list)
        return forecast_matrix
   

    def get_name(self):
        return self._name
    
    
    def get_model(self):
        return self._models
    
    
    def set_tour(self, X_tour, df_X_tour, X_embedding_tour):
        self._X_tour = X_tour
        self._df_X_tour = df_X_tour
        self._X_embedding_tour = X_embedding_tour

        
    def del_data(self):
        del self._X_tour
        del self._window_size
        del self._df_all_metadata
        del self._df_X_tour
        del self._all_data_matrix
        del self._museum_sequence_path 
        
        del self._split_time
        del self._X
        del self._batch_size
        del self._shuffle_buffer_size 
        
        del self._forecast_matrix
        del self._df_predicted_tour
        del self._predicted_code_list