from abc import ABC, abstractmethod


class Sequence_generator_class(ABC):
    
    @abstractmethod
    def predict_tour(self):
        pass
    
    @abstractmethod
    def get_predicted_tour_matrix(self):
        pass
    
    @abstractmethod
    def get_name(self):
        pass
    
    @abstractmethod
    def del_data(self):
        pass
    
    @abstractmethod
    def set_tour(self, X_tour, df_X_tour, X_embedding_tour):
        pass
    

    def get_tour(self):
        return self._X_tour

