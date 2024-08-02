## GCA with Multilinear Polynomials of degree up to 3
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

random_matrix = np.random.rand(2*n, 2*n)
Q, R = np.linalg.qr(random_matrix)

X = np.zeros([2*n,N])
for i in range(2*n):
    X = X + np.dot(np.array([Q[:,i]]).T,np.array([w[i]]))


# Functions
def generate_unique_pair():
    # Generate the first number
    num1 = random.randint(0, 14)
    
    # Generate the second number and ensure it is not equal to the first number
    while True:
        num2 = random.randint(0, 14)
        if num2 != num1:
            break
    
    return (num1, num2)
	
# hyperparameters
epsilon = 0.00001
n_correct = 0

# Feature Extraction
d_j = X.copy()
sigma_j = np.cov(d_j)
s = np.dot(Q.T, np.dot(sigma_j, Q))
s_diagonal = s.diagonal()
E_j = np.where(s_diagonal < epsilon)[0]
X_bar = X.copy()
for i in E_j:
    X_bar = X_bar - np.dot(np.dot(X.T, np.array([Q[:,i]]).T), np.array([Q[:,i]])).T

sigma_j_bar = np.cov(X_bar)
s_bar = np.dot(Q.T, np.dot(sigma_j_bar, Q))
s_diagonal_bar = s_bar.diagonal()
indices = np.argsort(s_diagonal_bar)[-2:]
l_j = indices[np.argsort(s_diagonal_bar[indices])[::-1]]
L_j = l_j
E_j = L_j
    

z_1 = np.dot(np.array(Q[:,L_j[0]]),X)
z_1_hat = z_1/(np.std(z_1))

z_2 = np.dot(np.array(Q[:,L_j[1]]),X)
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
for j in range(13):
    s = np.dot(Q.T, np.dot(sigma_j, Q))
    s_diagonal = s.diagonal()
    e_j = np.where(s_diagonal < epsilon)[0]
    e_j_unique_elements = np.setdiff1d(e_j, E_j)
    E_j = np.concatenate((E_j, e_j_unique_elements))
    X_bar = X.copy()
    for i in E_j:
        X_bar = X_bar - np.dot(np.dot(X.T, np.array([Q[:,i]]).T), np.array([Q[:,i]])).T
        
    sigma_j_bar = np.cov(X_bar)
    s_bar = np.dot(Q.T, np.dot(sigma_j_bar, Q))
    s_diagonal_bar = s_bar.diagonal()
    l_j = np.argmax(s_diagonal_bar)
    L_j = np.append(L_j, l_j)
    z_j = np.dot(np.array(Q[:,l_j]),X)
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
        
    sigma_j = np.cov(d_j)
    eigenvalues_sigma_j, _ = np.linalg.eig(sigma_j)
    max_eigenvalues_sigma_j = np.max(eigenvalues_sigma_j)

new_array = np.arange(15)
if np.array_equal(L_j, new_array):
    n_correct = n_correct + 1
    print('Number of correct extracted features:',n_correct)
else:
    if np.sum(L_j)==105:
        n_correct = n_correct + 1
        print('Number of correct extracted features:',n_correct)
    else:
        print('Number of correct extracted features:',n_correct, ', L_j =', L_j, ', E_j =', E_j, ', e_j =', e_j)
