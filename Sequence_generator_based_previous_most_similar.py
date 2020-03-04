import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,euclidean_distances
from Sequence_generator_models import Sequence_generator_class

class Sequence_generator_based_previous_most_similar(Sequence_generator_class):
    
    def __init__(self, window_size, df_all_metadata, all_data_matrix):
        self._name = "Sequence generator based previous most similar"
        self._window_size = window_size
        self._df_all_metadata = df_all_metadata
        self._all_data_matrix = all_data_matrix
        
        
    def _get_sim_matrix(self, code, all_data_matrix):
        #get the mean vector
        mean_code = np.mean(code, axis=0)
        mean_code.shape

        #Find most similar
        return cosine_similarity(mean_code.reshape((1,-1)), all_data_matrix)
    

    def _get_artwork_index(self, sim_matrix):
    
        #Sort indexes
        sort_index = np.argsort(sim_matrix.reshape((-1,)))[-300:]

        sort_index = np.flip(sort_index)
        #Find most similar artwork index with random walk
        sim_artwork_index = np.random.choice(sort_index, 2, replace=False)[0]

        if np.isclose(sim_matrix[:,sim_artwork_index][0], 1.):
            #Because the top is the current artwork
            return sort_index[-1]
        else:
            return sim_artwork_index
    
    
    def _drop_selected_artwork(self, indexes, df_all_metadata, all_data_matrix):
    
        #Remove from metadata
        df_removed = df_all_metadata.copy()
        df_removed = df_removed.drop(indexes)
        df_removed = df_removed.reset_index(drop=True)


        #Remove ftom code matrix
        code_matrix = all_data_matrix.copy()
        code_matrix = np.delete(code_matrix, indexes, 0)

        return df_removed, code_matrix
   

    def _find_artworks_indexes(self, i, window_size, df_all_metadata, df_X_tour):
    
        indexes = []
        for j in range(window_size):
            row = df_all_metadata[(df_all_metadata['author']==df_X_tour.iloc[i+j]['author']) & (df_all_metadata['title']==df_X_tour.iloc[i+j]['title'])]

            #Because the artwork may be already deleted
            if row.shape[0] != 0:
                indexes.append(row.index[0])

        return indexes
    
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
        
        #Check if window size is bigger than the tour length
        for i in range(self._X_tour.shape[0] - self._window_size):

            #Get current codes
            code = self._X_tour[i : i + self._window_size, :]

            index_artwork_selected = self._find_artworks_indexes(i, self._window_size, df_all_metadata, self._df_X_tour)

            df_current_metadata, current_data_matrix = self._drop_selected_artwork(index_artwork_selected, df_all_metadata, all_data_matrix)


            #Find most similar
            sim_matrix = self._get_sim_matrix(code, current_data_matrix)

            #Find most similar artwork index
            sim_artwork_index = self._get_artwork_index(sim_matrix)

            #Save in dataframe 
            self._df_predicted_tour = self._df_predicted_tour.append({
                            'title' : df_all_metadata.iloc[sim_artwork_index]['title'],
                            'author': df_all_metadata.iloc[sim_artwork_index]['author'],
                            'tour_path': df_all_metadata.iloc[sim_artwork_index]['tour_path'],
                            'image_url': df_all_metadata.iloc[sim_artwork_index]['image_url'],
                            'sim_value': sim_matrix[:,sim_artwork_index][0]
                           }, 
                           ignore_index=True)

            #Save predicted code
            self._predicted_code_list.append(all_data_matrix[sim_artwork_index])

            #Remove chosen artwork
            df_all_metadata, all_data_matrix = self._drop_selected_artwork(sim_artwork_index, df_all_metadata, all_data_matrix)

        return self._df_predicted_tour
    
    
    def get_predicted_tour_matrix(self):
        #No tour predicted because the window size was too big
        if len(self._predicted_code_list) == 0:
            return np.array([])
        forecast_matrix = np.stack(self._predicted_code_list)
        return forecast_matrix
    
    
    def set_tour(self, X_tour, df_X_tour):
        self._X_tour = X_tour
        self._df_X_tour = df_X_tour

    
    def get_name(self):
        return self._name
    
    def del_data(self):
        del self._X_tour
        del self._window_size
        del self._df_all_metadata
        del self._df_X_tour
        del self._all_data_matrix
    