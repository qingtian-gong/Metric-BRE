import numpy as np
import scipy
import scipy.linalg
import numpy.random
import time

A = np.load('A.npy')
G = scipy.linalg.sqrtm(A)
b = 8
d = 2048

start = time.clock()
r = numpy.random.normal(0, 1, b*d).reshape(d, b)
rG = r.T.dot(G)

labels = []
for line in open('training.txt'):
    tmp = line.split('\t')
    name, label = tmp[0].strip(), tmp[1].strip()
    labels.append(label)

X = np.load('encoded_images_train.npy')
Gx = G.dot(X.T)
chk = rG.dot(X.T)
n = X.shape[0]

h = {}
for i in range(n):
    s = 0
    for j in range(b):
        s *= 2
        if chk[j, i] >= 0:
            s += 1
    h.setdefault(s, [])
    h[s].append(i)

test_labels = []
for line in open('testing.txt'):
    tmp = line.split('\t')
    name, label = tmp[0].strip(), tmp[1].strip()
    test_labels.append(label)

middle = time.clock()
X_test = np.load('encoded_images_test.npy')
n_test = X_test.shape[0]
chk_test = rG.dot(X_test.T)

import heapq
nRank = 50
results = np.zeros(nRank)
for i in range(n_test):
    s = 0
    for j in range(b):
        s *= 2
        if chk_test[j, i] >= 0:
            s += 1
    if not s in h:
        continue
    heap = []
    Gx_test = G.dot(X_test[i].T)
    for j in h[s]:
        dA = np.linalg.norm(Gx_test - Gx[:,j])
        if labels[j] == test_labels[i]:
            heap.append((dA, j, 1))
        else:
            heap.append((dA, j, 0))
    j = 0
    heapq.heapify(heap)
    while heap and j < nRank:
        d, k, f = heapq.heappop(heap)
        results[j] += f
        j += 1
print 'Time for hashing dataset (s):', middle - start
print 'Time for querying all tests (s):', time.clock() - middle
fout = open('stat_MetricLSH.txt','w')
for i in range(nRank):
    if i > 0:
        results[i] += results[i-1]
    print >>fout, '%d\t%d\t%f'%(i, results[i], results[i] * 1.0 / n_test / (i + 1))
fout.close()
