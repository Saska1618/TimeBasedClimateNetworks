import pandas as pd
import numpy as np
from dtaidistance import dtw

def calculate_similarity(month1_data, month2_data, weights={'deriv': 0.7, 'tg': 0.1, 'tn': 0.1, 'tx': 0.1}):
    '''
    Calculates the similarity between two months based on their climate data.
    
    Args:
        month1_data (dict): Data for the first month.
        month2_data (dict): Data for the second month.
        weights (dict): Weights for combining the similarity scores.
        
    Returns:
        float: A similarity score between 0 and 1 (1 being most similar).
    '''
    # 1. Similarity of tg_derivatives using DTW
    d1 = np.array(month1_data['tg_derivatives'], dtype=np.double)
    d2 = np.array(month2_data['tg_derivatives'], dtype=np.double)
    dtw_distance = dtw.distance(d1, d2)
    # A simple normalization for DTW distance to convert it to a similarity score
    sim_deriv = 1 / (1 + dtw_distance)
    
    # 2. Similarity of mean_tn and mean_tx
    sim_tg = 1 / (1 + abs(month1_data['mean_tn'] - month2_data['mean_tn']))
    sim_tn = 1 / (1 + abs(month1_data['mean_tn'] - month2_data['mean_tn']))
    sim_tx = 1 / (1 + abs(month1_data['mean_tx'] - month2_data['mean_tx']))
    
    # 3. Combine similarities with weights
    total_similarity = (weights['deriv'] * sim_deriv + 
                        weights['tg'] * sim_tg + 
                        weights['tn'] * sim_tn + 
                        weights['tx'] * sim_tx)
    
    return total_similarity