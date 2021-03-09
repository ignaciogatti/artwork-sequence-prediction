import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns # This is for visualization

def matrix_sqrt(x):
    '''
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a numpy matrix
    '''
    y = sqrtm(x)
    return y


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features) 
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    '''
    return np.linalg.norm(mu_x - mu_y) + np.trace(sigma_x + sigma_y - 2 * matrix_sqrt(sigma_x@sigma_y))



def compute_fretech_distance(forecast_feature_matrix, real_feature_matrix):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians.
    Parameters:
        forecast_matrix: features predicted by the model
        real_matrix: real values for the features
    '''
    
    mu_forecast_feature = np.mean(forecast_feature_matrix, axis=0)
    sigma_forecast_feature = np.cov(forecast_feature_matrix, rowvar=False)

    mu_real_feature = np.mean(real_feature_matrix, axis=0)
    sigma_real_feature = np.cov(real_feature_matrix, rowvar=False)
    
    return frechet_distance(
        mu_x=mu_forecast_feature,
        mu_y=mu_real_feature,
        sigma_x=sigma_forecast_feature,
        sigma_y=sigma_real_feature)