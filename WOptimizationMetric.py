import numpy as np
import scipy
import scipy.linalg
import numpy.random
import time

b = 8
d = 2048

labels = []
for line in open('training.txt'):
    tmp = line.split('\t')
    name, label = tmp[0].strip(), tmp[1].strip()
    labels.append(label)

X = np.load('encoded_images_train.npy')
n = X.shape[0]
norm_train = np.linalg.norm(X, axis = 1)
X = X / norm_train.reshape(n, 1)

test_labels = []
for line in open('testing.txt'):
    tmp = line.split('\t')
    name, label = tmp[0].strip(), tmp[1].strip()
    test_labels.append(label)

X_test = np.load('encoded_images_test.npy')
n_test = X_test.shape[0]
norm_test = np.linalg.norm(X_test, axis = 1)
X_test = X_test / norm_test.reshape(n_test, 1)

d = {}
constraints = np.load('constraints.npy')

for i, j, s in constraints:
    dij = 0.5 * np.linalg.norm(X[i,] - X[j,]) ** 2
    d.setdefault(i, {})
    d[i][j] = dij
    d.setdefault(j, {})
    d[j][i] = dij

A = np.load('A_normal.npy')
inner_prod = X.dot(A).dot(X.T)

W = numpy.random.normal(0, 1, b * n)
W = W.reshape(b, n)

start = time.clock()
c = W.dot(inner_prod)
i = 0
changed = True
while i < 10 and changed:
    changed = False
    for p in range(b):
        q = numpy.random.randint(n)
        t = []
        for j in range(n):
            t.append((j, W[p, q] - c[p, j] * 1.0 / inner_prod[q, j]))
        t.sort(key = lambda x:x[1])
        Wpg_bak = W[p,q]
        Wpq_tmp = t[0][1] / 2.0
        W_tmp = W.copy()
        W_tmp[p, q] = Wpq_tmp
        W[p,q] = Wpq_tmp
        O = 0
        for i1, i2, s in constraints:
            dnorm = np.linalg.norm(np.where(W.dot(inner_prod[:,i1]) >= 0, 1, 0) - np.where(W.dot(inner_prod[:,i2]) >= 0, 1, 0))
            d_tilde = dnorm ** 2 / b
            O += (d[i1][i2] - d_tilde) ** 2
        O0 = O
        for j in range(1, n):
            W_new = W.copy()
            W_new[p,q] = (t[j-1][1] + t[j][1]) / 2.0
            deltaO = 0
            if t[j-1][0] not in d:
                W_tmp = W_new.copy()
                continue
            for k in d[t[j-1][0]]:
                dnorm = np.linalg.norm(np.where(W_tmp.dot(inner_prod[:,k]) >= 0, 1, 0) - np.where(W_tmp.dot(inner_prod[:,t[j-1][0]]) >= 0, 1, 0))
                d_tilde = dnorm ** 2 / b
                deltaO -= (d[t[j-1][0]][k] - d_tilde) ** 2
                dnorm = np.linalg.norm(np.where(W_new.dot(inner_prod[:,k]) >= 0, 1, 0) - np.where(W_new.dot(inner_prod[:,t[j-1][0]]) >= 0, 1, 0))
                d_tilde = dnorm ** 2 / b
                deltaO += (d[t[j-1][0]][k] - d_tilde) ** 2
            if O0 + deltaO < O:
                W[p,q] = (t[j-1][1] + t[j][1]) / 2.0
                O = O0 + deltaO
            O0 += deltaO
            W_tmp = W_new.copy()
        if abs(W[p,q] - Wpg_bak) > 0.001:
            changed = True
            for j in range(n):
                c[p,j] += (W[p,q] - Wpg_bak) * inner_prod[j, q]

    O = 0
    for i1, i2, s in constraints:
        dnorm = np.linalg.norm(np.where(W.dot(inner_prod[:,i1]) >= 0, 1, 0) - np.where(W.dot(inner_prod[:,i2]) >= 0, 1, 0))
        d_tilde = dnorm ** 2 / b
        O += (d[i1][i2] - d_tilde) ** 2
    i += 1
print 'Time for learning W (s):', time.clock() - start
np.save('W_metric.npy', W)

