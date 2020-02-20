from Sequence_generator_models import Sequence_generator_class
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
import tensorflow as tf
from Prediction_model_feature import Prediction_model_feature
from IPython.display import clear_output
import numpy as np
import pandas as pd
import os


class Sequence_generator_rnn(Sequence_generator_class):
    
    def __init__(self, window_size, X_tour, df_X_tour, df_all_metadata, all_data_matrix, museum_sequence_path, batch_size, shuffle_buffer_size, X, split_time):
        self._name= "Sequence_generator_rnn"
        self._X_tour = X_tour
        self._window_size = window_size
        self._df_all_metadata = df_all_metadata
        self._df_X_tour = df_X_tour
        self._all_data_matrix = all_data_matrix
        self._museum_sequence_path = museum_sequence_path
        
        self._split_time = split_time
        self._X = X
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        
        self.models = self._load_model()
        
        
    def _load_weights(self, model, index, museum_sequence_path):
        #Find the folder where the weights are saved
        model_feature_folder = os.path.join(museum_sequence_path['weights_folder'], 'model_feature_'+str(index))    
        #Load weights
        model.load_weights(os.path.join(model_feature_folder, 'weights_feature_'+str(index)))
        return model
    
    
    def _load_model(self):
        self._n_features = self._X.shape[1]
        self._models = []
        for i in range(self._n_features):
            clear_output(wait=True)
            #Create model
            model_prediction = Prediction_model_feature(
                X=self._X[:, 0],
                split_time=self._split_time,
                train_batch_size=self._batch_size, 
                val_batch_size=self._batch_size, 
                window_size=self._window_size, 
                shuffle_buffer=self._shuffle_buffer_size,
                name="feature " + str(0))
            model = model_prediction.get_model()
            #Load weights
            model =self._load_weights(model, i, self._museum_sequence_path)
            self._models.append(model)
            
        return self._models
    
    
    def _model_forecast(self, model, series, window_size, batch_size):
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
    

    def _predict_features(self):
        
        predicted_features = []
        for feature in range(self._n_features):
            #Predict feature i
            x_feature = self._X_tour[:,feature]
            rnn_forecast = self._model_forecast(self._models[feature], x_feature, self._window_size, self._batch_size)
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
        forecast_matrix = np.stack(self._predicted_code_list)
        return forecast_matrix
   

    def get_name(self):
        return self._name
    
    def get_model(self):
        return self._models