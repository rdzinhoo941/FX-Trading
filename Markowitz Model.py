import pandas as pd
import numpy as np
import random as rd


# We don't have yet the correlation matrix of the currencies so for the moment we will
# work with a random matrix of dimension 3, to kept the coherence each coefficients of the 
# matrix will remains in the intevral [0,1] has each coef represent the covariance between
# two currency

M = np.eye(3)  # Crée une matrice avec des 1 sur la diagonale (Section 2.4.1)

for i in range(3):
    for j in range(i + 1, 3):
        r = rd.uniform(-0.5, 0.8)  
        M[i, j] = r
        M[j, i] = r  

print("Matrice de corrélation simulée (3x3) :")
print(M)

#same for the expected return, we will use a random vector of dimension 3

E=np.zeros(3)
for i in range(3):
    E[i]=rd.randint(-10,10)
print(E)

