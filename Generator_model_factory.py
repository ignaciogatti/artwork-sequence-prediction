from Sequence_generator_based_previous_most_similar import Sequence_generator_based_previous_most_similar
from Sequence_generation_rnn import Sequence_generator_rnn
from Sequence_generation_rnn_multivariate import Sequence_generator_rnn_multivariate
from abc import ABC, abstractmethod
import tensorflow as tf


class Generator_model_factory(ABC):
    
    @abstractmethod
    def get_model(self):
        pass
    
    

class Generator_model_most_similar(Generator_model_factory):
    
    def __init__(self, window_size, all_data_matrix, df_all_metadata):
        self._window_size = window_size
        self._all_data_matrix = all_data_matrix
        self._df_all_metadata = df_all_metadata
        
        
        
    def get_model(self):

        seq_generator_most_sim = Sequence_generator_based_previous_most_similar( 
            window_size = self._window_size, 
            all_data_matrix = self._all_data_matrix, 
            df_all_metadata = self._df_all_metadata )

        return seq_generator_most_sim
    
    
    def __str__(self):
        return 'generated_sequence_based_previous_most_similar'
    
    
class Generator_model_rnn(Generator_model_factory):
    
    def __init__(self, X, window_size, all_data_matrix, df_all_metadata, museum_sequence_path, batch_size, shuffle_buffer_size, split_time):
        
        self._X = X
        self._window_size = window_size
        self._all_data_matrix = all_data_matrix
        self._df_all_metadata = df_all_metadata
        self._museum_sequence_path = museum_sequence_path
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._split_time = split_time
        
    
        
    def get_model(self):
        #Clear all variables from previous model
        tf.keras.backend.clear_session()
    
        seq_generator_rnn = Sequence_generator_rnn(
            X = self._X, 
            all_data_matrix = self._all_data_matrix, 
            df_all_metadata = self._df_all_metadata, 
            batch_size = self._batch_size, 
            window_size = self._window_size, 
            split_time = self._split_time, 
            museum_sequence_path = self._museum_sequence_path, 
            shuffle_buffer_size = self._shuffle_buffer_size )

        return seq_generator_rnn
    
    
    def __str__(self):
        return 'generated_sequence_rnn'
    

class Generator_model_rnn_multivariate(Generator_model_factory):
    
    def __init__(self, X, window_size, all_data_matrix, df_all_metadata, museum_sequence_path, batch_size, shuffle_buffer_size, split_time, n_influence_features):
        
        self._X = X
        self._window_size = window_size
        self._all_data_matrix = all_data_matrix
        self._df_all_metadata = df_all_metadata
        self._museum_sequence_path = museum_sequence_path
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._split_time = split_time
        self._n_influence_features = n_influence_features
        
    
        
    def get_model(self):
        #Clear all variables from previous model
        tf.keras.backend.clear_session()
    
        seq_generator_rnn_multi = Sequence_generator_rnn_multivariate(
            X = self._X, 
            all_data_matrix = self._all_data_matrix, 
            df_all_metadata = self._df_all_metadata, 
            batch_size = self._batch_size, 
            window_size = self._window_size, 
            split_time = self._split_time, 
            museum_sequence_path = self._museum_sequence_path, 
            shuffle_buffer_size = self._shuffle_buffer_size,
            n_influence_features = self._n_influence_features)

        return seq_generator_rnn_multi
    
    
    def __str__(self):
        return 'generated_sequence_rnn_multivariate'