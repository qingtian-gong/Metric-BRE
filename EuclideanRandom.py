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

start = time.clock()

h = {}
for i in range(n):
    s = np.random.randint(2 ** b)
    h.setdefault(s, [])
    h[s].append(i)


test_labels = []
for line in open('testing.txt'):
    tmp = line.split('\t')
    name, label = tmp[0].strip(), tmp[1].strip()
    test_labels.append(label)

X_test = np.load('encoded_images_test.npy')
n_test = X_test.shape[0]

middle = time.clock()

import heapq
nRank = 50
results = np.zeros(nRank)
for i in range(n_test):
    s = np.random.randint(2 ** b)
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
    tmpF = 0
    while heap and j < nRank:
        d, k, f = heapq.heappop(heap)
        results[j] += f
        tmpF += f
        j += 1
print 'Time for hashing dataset (s):', middle - start
print 'Time for querying all tests (s):', time.clock() - middle
fout = open('stat_EuclideanRandom.txt', 'w')
for i in range(nRank):
    if i > 0:
        results[i] += results[i-1]
    print >>fout, '%d\t%d\t%f'%(i, results[i], results[i] * 1.0 / n_test / (i + 1))
fout.close()
