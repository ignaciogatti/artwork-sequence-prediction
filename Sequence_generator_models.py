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

