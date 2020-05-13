from Sequence_generator_based_previous_most_similar import Sequence_generator_based_previous_most_similar
from Sequence_generation_rnn import Sequence_generator_rnn
from Sequence_generation_rnn_multivariate import Sequence_generator_rnn_multivariate
from Sequence_generation_rnn_embedding import Sequence_generator_rnn_embedding
from Sequence_generation_rnn_code_embedding import Sequence_generator_rnn_code_embedding
from abc import ABC, abstractmethod
import tensorflow as tf
import os


class Generator_model_factory(ABC):
    
    @abstractmethod
    def get_model(self):
        pass

'''
####################  Generator Most similar model ##############################
'''


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
    
'''
####################  Generator RNN models ##############################
'''
    
class Abstract_Generator_model_rnn(Generator_model_factory):
    
    def __init__(self, X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, conv_filter=20, lstm_filter=40, dense_filter=16, prediction_length=1):
        
        self._X = X
        self._window_size = window_size
        self._all_data_matrix = all_data_matrix
        self._df_all_metadata = df_all_metadata
        self._CONFIG_PATH = CONFIG_PATH
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._split_time = split_time
        self._conv_filter = conv_filter
        self._lstm_filter = lstm_filter
        self._dense_filter = dense_filter
        self._prediction_length = prediction_length
    
    
    def _get_trained_weights_path(self):
        trained_weights_path = {
            'weights_folder' : os.path.join(self._CONFIG_PATH, 'config_'+str(self._window_size)+'/trained_model_weights')
        }
        return trained_weights_path

    
    
class Generator_model_rnn(Abstract_Generator_model_rnn):
    
    def __init__(self, X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, conv_filter=20, lstm_filter=40, dense_filter=16, prediction_length=1):
        super().__init__(X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, conv_filter, lstm_filter, dense_filter, prediction_length)
            
        
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
            museum_sequence_path = self._get_trained_weights_path(), 
            shuffle_buffer_size = self._shuffle_buffer_size,
            conv_filter=self._conv_filter, 
            lstm_filter=self._lstm_filter, 
            dense_filter=self._dense_filter, 
            prediction_length=self._prediction_length)

        return seq_generator_rnn
    
    
    def __str__(self):
        return 'generated_sequence_rnn'
    

class Generator_model_rnn_multivariate(Abstract_Generator_model_rnn):
    
    def __init__(self, X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, n_influence_features, conv_filter=20, lstm_filter=40, dense_filter=16, prediction_length=1):
        super().__init__(X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, conv_filter, lstm_filter, dense_filter, prediction_length)
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
            museum_sequence_path = self._get_trained_weights_path(), 
            shuffle_buffer_size = self._shuffle_buffer_size,
            n_influence_features = self._n_influence_features,
            conv_filter=self._conv_filter, 
            lstm_filter=self._lstm_filter, 
            dense_filter=self._dense_filter, 
            prediction_length=self._prediction_length)

        return seq_generator_rnn_multi
    
    
    def __str__(self):
        return 'generated_sequence_rnn_multivariate'
    
    
class Generator_model_rnn_embedding(Abstract_Generator_model_rnn):
    
    def __init__(self, X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, X_embedding, conv_filter=20, lstm_filter=40, dense_filter=16, prediction_length=1):
        
        super().__init__(X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, conv_filter, lstm_filter, dense_filter, prediction_length)
        self._X_embedding = X_embedding
        
    
        
    def get_model(self):
        #Clear all variables from previous model
        tf.keras.backend.clear_session()
    
        seq_generator_rnn_embedding = Sequence_generator_rnn_embedding(
            X = self._X, 
            all_data_matrix = self._all_data_matrix, 
            df_all_metadata = self._df_all_metadata, 
            batch_size = self._batch_size, 
            window_size = self._window_size, 
            split_time = self._split_time, 
            museum_sequence_path = self._get_trained_weights_path(), 
            shuffle_buffer_size = self._shuffle_buffer_size,
            X_embeddings = self._X_embedding,
            conv_filter=self._conv_filter, 
            lstm_filter=self._lstm_filter, 
            dense_filter=self._dense_filter, 
            prediction_length=self._prediction_length)

        return seq_generator_rnn_embedding
    
    
    def __str__(self):
        return 'generated_sequence_rnn_embedding'
    
    
    
class Generator_model_rnn_code_embeding(Abstract_Generator_model_rnn):
    
    def __init__(self, X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, conv_filter=20, lstm_filter=40, dense_filter=16, prediction_length=1):
        super().__init__(X, window_size, all_data_matrix, df_all_metadata, CONFIG_PATH, batch_size, shuffle_buffer_size, split_time, conv_filter, lstm_filter, dense_filter, prediction_length)
            
        
    def get_model(self):
        #Clear all variables from previous model
        tf.keras.backend.clear_session()
    
        seq_generator_rnn_code_emb = Sequence_generator_rnn_code_embedding(
            X = self._X, 
            all_data_matrix = self._all_data_matrix, 
            df_all_metadata = self._df_all_metadata, 
            batch_size = self._batch_size, 
            window_size = self._window_size, 
            split_time = self._split_time, 
            museum_sequence_path = self._get_trained_weights_path(), 
            shuffle_buffer_size = self._shuffle_buffer_size,
            conv_filter=self._conv_filter, 
            lstm_filter=self._lstm_filter, 
            dense_filter=self._dense_filter, 
            prediction_length=self._prediction_length)

        return seq_generator_rnn_code_emb
    
    
    def __str__(self):
        return 'generated_sequence_rnn_code_embedding'