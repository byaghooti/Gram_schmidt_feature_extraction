## GFA with Multilinear Polynomials of degree up to 3
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import random

# Imoprt Dataset
N = 1000
n = 15
w = np.zeros([2*n,N])
for i in range(n):
    w[i,:] = np.random.normal(loc=0, scale=np.sqrt(n-i), size=N)
w = w - w.mean(axis=1, keepdims=True)

for i in range(n,2*n):
    pair = generate_unique_pair()
    redundant = w[pair[0]]*w[pair[1]]
    var_redundant = np.var(w[pair[0]]*w[pair[1]])
    var_min = min(np.var(w[pair[0]]),np.var(w[pair[1]]))
    w[i,:] = np.sqrt(0.85*var_min/var_redundant)*redundant
w = w - w.mean(axis=1, keepdims=True)
X = w.copy()


# Functions
def generate_unique_pair():
    # Generate the first number
    num1 = random.randint(0, n-1)
    
    # Generate the second number and ensure it is not equal to the first number
    while True:
        num2 = random.randint(0, n-1)
        if num2 != num1:
            break
    
    return (num1, num2)
	
# hyperparameters
epsilon = 0.00001
n_correct = 0

# Feature Selection
d_j = X.copy()
sigma_j = np.var(d_j,axis=1)
E_j = np.where(sigma_j < epsilon)[0]
indices_of_largest_elements = np.argpartition(sigma_j, -2)[-2:]
s_j = indices_of_largest_elements[np.argsort(-sigma_j[indices_of_largest_elements])]
S_j = s_j
E_j = S_j
z_1 = X[S_j[0]]
z_1_hat = z_1/(np.std(z_1))
z_2 = X[S_j[1]]
z_2_hat = z_2/(np.std(z_2))
F_hat = np.array([z_1_hat, z_2_hat])
H_1 = np.array([z_1, z_2])
z_12 = z_1*z_2 - np.mean(z_1*z_2)
H_2 = np.array([z_12])
H_2_j = H_2
z_12_tilde = z_12
for i in range(len(F_hat)):
    z_12_tilde = z_12_tilde - np.mean(z_12*F_hat[i])*F_hat[i]

z_12_hat = z_12_tilde/(np.std(z_12_tilde))
F_hat = np.append(F_hat, np.array([z_12_hat]), axis=0)
F_j_hat = np.array([z_12_hat])

for i in range(len(F_hat)):
    d_j = d_j - np.matmul(np.array([np.mean(X*F_hat[i],axis=1)]).T, np.array([F_hat[i]]))

sigma_j = np.cov(d_j)

for j in range(n-2):
    sigma_j = np.var(d_j,axis=1)
    e_j = np.where(sigma_j < epsilon)[0]
    e_j_unique_elements = np.setdiff1d(e_j, E_j)
    E_j = np.concatenate((E_j, e_j_unique_elements))

    mask = np.ones(len(sigma_j), dtype=bool)
    mask[E_j] = False
    max_index = np.argmax(sigma_j[mask])
    s_j = np.where(mask)[0][max_index]
    S_j = np.append(S_j, s_j)
    z_j = X[s_j]
    z_j_tilde = z_j
    for F_hat_ind in range(len(F_hat)):
        z_j_tilde = z_j_tilde - np.mean(z_j*F_hat[F_hat_ind])*F_hat[F_hat_ind]
    z_j_hat = z_j_tilde/(np.std(z_j_tilde))
    F_hat = np.append(F_hat, np.array([z_j_hat]), axis=0)
    F_j_hat = np.array([z_j_hat])
    H_2_j = np.empty((len(H_1), N))
    H_2_j[:] = np.nan
    for H_1_ind in range(len(H_1)):
        z_j_2 = z_j*H_1[H_1_ind] - np.mean(z_j*H_1[H_1_ind])
        H_2_j[H_1_ind] = z_j_2
        z_j_2_tilde = z_j_2
        for F_hat_ind in range(len(F_hat)):
            z_j_2_tilde = z_j_2_tilde - np.mean(z_j_2*F_hat[F_hat_ind])*F_hat[F_hat_ind]
        z_j_2_hat = z_j_2_tilde/(np.std(z_j_2_tilde))
        F_hat = np.append(F_hat, np.array([z_j_2_hat]), axis=0)
        F_j_hat = np.append(F_j_hat, np.array([z_j_2_hat]), axis=0)
    H_1 = np.append(H_1, np.array([z_j]), axis=0)

    for i in range(len(F_j_hat)):
        d_j = d_j - np.matmul(np.array([np.mean(X*F_j_hat[i],axis=1)]).T, np.array([F_j_hat[i]]))
        

new_array = np.arange(n)
if np.array_equal(S_j, new_array):
    n_correct = n_correct + 1
    print('Number of correct selected features:',n_correct)
else:
    if np.sum(S_j)==(n*(n-1)/2):
        n_correct = n_correct + 1
        print('Number of correct selected features:',n_correct)
    else:
        print('Number of correct selected features:',':',n_correct, ', S_j =', S_j, ', E_j =', E_j, ', e_j =', e_j)
