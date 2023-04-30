import numpy as np

def givens_qr(R_I, n, m):
    rows = list(range(m+n, m, -1))
    l = 1
    Q = np.eye(n+m)
    for k in rows:
        for i in list(range(min(l,n), 0, -1)):
            R_I, Q_1 = givens_rotation(R_I, k-1, n-i)
            Q = np.dot(Q_1, Q)
        l += 1
    return R_I, Q.T

def givens_rotation(A, i, j):
    n,m = A.shape
    r = (A[j,j]**2 + A[i,j]**2)**(0.5)
    c = A[j,j]/r
    s = A[i,j]/r
    Q = np.eye(n)
    Q[j,j] = c
    Q[i,i] = c
    Q[i,j] = -s
    Q[j,i] = s
    
    B = A.astype(float)
    S = np.array([[c,s], [-s,c]])
    C = A[(j, i),:]
    D = np.dot(S, C)
    B[j] = D[0]
    B[i] = D[1] 
    return B, Q