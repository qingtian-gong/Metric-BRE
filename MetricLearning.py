import numpy as np
import time
X = np.load('encoded_images_train.npy')
u, l = 1, 5
n = X.shape[0]
A0 = np.cov(X.T)
A = A0

constraints = np.load('constraints.npy')

start = time.clock()

for i, j, s in constraints:
        dx = (X[i] - X[j]).reshape([2048, 1])
        dA = dx.T.dot(A).dot(dx)[0,0]
        if s == 1:
            alpha = 1.0 / dA - 1.0 / u
        else:
            alpha = 1.0 / l - 1.0 / dA
        if alpha >= 0:
            continue
        if s == 1:
            beta = alpha / (1 - alpha * dA)
        else:
            beta = -alpha / (1 + alpha * dA)
        A += A.dot(dx).dot(dx.T).dot(A) * beta

np.save('A.npy', A)
print 'Time for learning A (s):', time.clock() - start
