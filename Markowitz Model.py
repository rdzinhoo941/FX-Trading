import pandas as pd
import numpy as np
import random as rd
from scipy.optimize import minimize

M = np.eye(3)
for i in range(3):
    for j in range(i + 1, 3):
        r = rd.uniform(-0.5, 0.8)  
        M[i, j] = r
        M[j, i] = r 

E = np.array([rd.uniform(-0.1, 0.1) for _ in range(3)])

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

poids_optimaux = markowit_optimisation(E, M)

print("Vecteur E :")
print(E)
print("Poids optimaux :")
print(poids_optimaux)
print("Somme des poids :")
print(np.sum(poids_optimaux))