import numpy as np
from Sequence_generation_rnn import Sequence_generator_rnn



class Sequence_generator_rnn_code_embedding(Sequence_generator_rnn):
    
    def set_tour(self, X_tour, df_X_tour, X_embedding_tour):
        self._X_tour = np.hstack((X_tour, X_embedding_tour))
        self._df_X_tour = df_X_tour
        self._X_embedding_tour = X_embedding_tour
