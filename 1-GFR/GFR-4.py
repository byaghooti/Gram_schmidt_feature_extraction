## GFR with Multilinear Polynomials of degree up to 4
# Import Packages
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Imoprt Dataset
import scipy.io
mat = scipy.io.loadmat("COIL20.mat")
X = mat['X']
StanScal = StandardScaler()
X = StanScal.fit_transform(X)
X = X.T
d = X.shape[0]
N = X.shape[1]
print(d)
print(N)

# Functions
def norm(x):
    return np.sqrt(np.mean(x**2))
	
# Threshold
eps = 0.01

# Feature Extraction
d_j = X.copy()
pca = PCA(n_components=2)
pca.fit(X.T)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
v = np.array([eigenvectors[0], eigenvectors[1]])
z = np.array([np.matmul(v[0], X), np.matmul(v[1], X)])
z1_hat = z[0]/norm(z[0])
z2_hat = z[1]/norm(z[1])
F_hat = np.array([z1_hat, z2_hat])
H_1 = np.array([z[0], z[1]])
z12 = z[0]*z[1] - np.mean(z[0]*z[1])
H_2 = np.array([z12])
H_2_j = H_2
z12_tilde = z12
for i in range(len(F_hat)):
    z12_tilde = z12_tilde - np.mean(z12*F_hat[i])*F_hat[i]
z12_hat = z12_tilde/norm(z12_tilde)
F_hat = np.append(F_hat, np.array([z12_hat]), axis=0)
F_j_hat = np.array([z12_hat])

for i in range(len(F_hat)):
    d_j = d_j - np.matmul(np.array([np.mean(X*F_hat[i],axis=1)]).T, np.array([F_hat[i]]))

eigenval_j = eigenvalues[1]


j = 0
j = 0
while eigenval_j>eps:
    pca = PCA(n_components=1)
    pca.fit(d_j.T)
    eigenval_j = pca.explained_variance_
    eigenvec_j = pca.components_
    print(j, ':', eigenval_j)
    v_j = eigenvec_j[0]
    z_j = np.matmul(v_j, X)
    z = np.append(z, np.array([z_j]), axis=0)
    v = np.append(v, np.array([v_j]), axis=0)
    z_j_tilde = z_j
    for F_hat_ind in range(len(F_hat)):
        z_j_tilde = z_j_tilde - np.mean(z_j*F_hat[F_hat_ind])*F_hat[F_hat_ind]
    z_j_hat = z_j_tilde/norm(z_j_tilde)
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
        z_j_2_hat = z_j_2_tilde/norm(z_j_2_tilde)
        F_hat = np.append(F_hat, np.array([z_j_2_hat]), axis=0)
        F_j_hat = np.append(F_j_hat, np.array([z_j_2_hat]), axis=0)
    H_1 = np.append(H_1, np.array([z_j]), axis=0)

    H_3_j = np.empty((len(H_2), N))
    H_3_j[:] = np.nan
    for H_2_ind in range(len(H_2)):
        z_j_3 = z_j*H_2[H_2_ind]
        H_3_j[H_2_ind] = z_j_3
        z_j_3_tilde = z_j_3
        for F_hat_ind in range(len(F_hat)):
            z_j_3_tilde = z_j_3_tilde - np.mean(z_j_3*F_hat[F_hat_ind])*F_hat[F_hat_ind]
        z_j_3_hat = z_j_3_tilde/norm(z_j_3_tilde)
        F_hat = np.append(F_hat, np.array([z_j_3_hat]), axis=0)
        F_j_hat = np.append(F_j_hat, np.array([z_j_3_hat]), axis=0)
    H_2 = np.append(H_2, H_2_j, axis=0)

    if j==0:
        H_3 = H_3_j
    else:
        H_4_j = np.empty((len(H_3), N))
        H_4_j[:] = np.nan
        for H_3_ind in range(len(H_3)):
            z_j_4 = z_j*H_3[H_3_ind]
            H_4_j[H_3_ind] = z_j_4
            z_j_4_tilde = z_j_4
            for F_hat_ind in range(len(F_hat)):
                z_j_4_tilde = z_j_4_tilde - np.mean(z_j_4*F_hat[F_hat_ind])*F_hat[F_hat_ind]
            z_j_4_hat = z_j_4_tilde/norm(z_j_4_tilde)
            F_hat = np.append(F_hat, np.array([z_j_4_hat]), axis=0)
            F_j_hat = np.append(F_j_hat, np.array([z_j_4_hat]), axis=0)
        H_3 = np.append(H_3, H_3_j, axis=0)

    for i in range(len(F_j_hat)):
        d_j = d_j - np.matmul(np.array([np.mean(X*F_j_hat[i],axis=1)]).T, np.array([F_j_hat[i]]))

    j = j + 1
	
print("Directions:", v)
