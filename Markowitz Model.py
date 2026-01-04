import pandas as pd
import numpy as np
import random as rd
from scipy.optimize import minimize
import extraction_forex_assets

vec_returns, mat_corr = extraction_forex_assets.get_returns_and_covariance(extraction_forex_assets.pairs_list)


def objective_function(weight_vector, mat_corr):
    return weight_vector @ mat_corr @ weight_vector

def markowit_optimisation(vec_returns, mat_corr):
    n = len(vec_returns)
    initial_guess = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(mat_corr,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return result.x

poids_optimaux = markowit_optimisation(vec_returns, mat_corr)


print("Vecteur E :")
print(vec_returns)
print("Poids optimaux :")
print(poids_optimaux)
print("Somme des poids :")
print(np.sum(poids_optimaux))