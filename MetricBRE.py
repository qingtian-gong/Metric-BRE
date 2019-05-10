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

A = np.load('A_normal.npy')

inner_prod = X.dot(A).dot(X.T)

W = np.load('W_metric.npy')

start = time.clock()
chk = W.dot(inner_prod)
h = {}
for i in range(n):
    s = 0
    for j in range(b):
        s *= 2
        if chk[j, i] >= 0:
            s += 1
    h.setdefault(s, [])
    h[s].append(i)

middle = time.clock()
chk_test = X_test.dot(A).dot(X.T).dot(W.T)

import heapq
nRank = 50
results = np.zeros(nRank)
for i in range(n_test):
    s = 0
    for j in range(b):
        s *= 2
        if chk_test[i, j] >= 0:
            s += 1
    if not s in h:
        continue
    heap = []
    for j in h[s]:
        dij = np.linalg.norm(X_test[i,] - X[j,])
        if labels[j] == test_labels[i]:
            heap.append((dij, j, 1))
        else:
            heap.append((dij, j, 0))
    j = 0
    heapq.heapify(heap)
    while heap and j < nRank:
        d, k, f = heapq.heappop(heap)
        results[j] += f
        j += 1
print 'Time for hashing dataset (s):', middle - start
print 'Time for querying all tests (s):', time.clock() - middle
fout = open('stat_MetricBRE.txt', 'w')
for i in range(nRank):
    if i > 0:
        results[i] += results[i-1]
    print >>fout, '%d\t%d\t%f'%(i, results[i], results[i] * 1.0 / n_test / (i + 1))
fout.close()
