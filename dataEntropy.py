import numpy as np
import matplotlib.pyplot as plt
from math import pi
from collections import namedtuple
np.random.seed(0)

#Features is a named tuple that contains the sets and their labels
Features = namedtuple('Features', ['sets', 'labels'])

def calculate_entropy(data):
    """Calculates the entropy for a one dimensional dataset
    Arguments:
    data -- vector of data
    Returns:
    entropy-- shannon entropy calculated with frequencies
    """
    hist = np.histogram(data, bins = 300)
    freqs = hist[0]
    pos_freqs = freqs[freqs>0].astype(float)
    norm_freqs  = pos_freqs/np.sum(pos_freqs)
    entropy = -np.sum(np.multiply(norm_freqs, np.log2(norm_freqs)))
    return(entropy)

def rotate(matrix, angle):
    """Rotates matrix given angle
    Arguments:
    matrix 
    angle -- angle in [0, pi]
    Returns:
    rotated_mat -- rotated matrix 
    """
    c,s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c,-s), (s,c)))
    rotated_mat = np.dot(np.dot(rotation,matrix), np.transpose(rotation))
    return rotated_mat

def generate_data(N):
    """
    Generates example to learn the entropy of Gaussian distributions. 
    Arguments: 
    N -- number of sets
    Returns:
    feats -- named tuple that contains sets and labels
    """
    #Selects a random covariance matrix
    cov_matrix = np.array([[0.5,0,],[0,10]])
    #Generates vector for rotation angles
    alpha_vec = np.random.uniform(low = 0, high = pi, size =N)

    #Generates list of rotated covariance matrices
    covariances = [rotate(cov_matrix, alpha_vec[i]) for i in range(N)]

    #Generates a sample X from N(0, cov_matrix)
    X = np.random.multivariate_normal(np.zeros(2), cov_matrix, 300)

    #Generates N sets by rotating the sample X 
    sets = [X.dot(covariances[i]) for i in range(N)]

    #Calculates the entropy of the marginal distribution 
    # of the first dimension of each set.    
    labels = [calculate_entropy(sets[i][:,0]) for i in range(N)]
    feats = Features(sets, labels)
    return feats
