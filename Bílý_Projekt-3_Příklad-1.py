import numpy as np


def lu_basic(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for j in range(n):
        if j == 0:
            v = A[j:, j]
        else:
            z = np.linalg.solve(L[0:j, 0:j], A[0:j, j])
            U[0:j, j] = z
            v = A[j:, j] - L[j:, 0:j] @ z

        if j < n - 1:
            L[j+1:n, j] = v[1:] / v[0]

        U[j, j] = v[0]

    return L, U


def lu_mn(A):
    m = A.shape[0]
    n = A.shape[1]
    L = np.eye(m)
    U = np.zeros((m, n))

    for j in range(n):
        if j == 0:
            v = A[j:, j]
        else:
            z = np.linalg.solve(L[0:j, 0:j], A[0:j, j])
            U[0:j, j] = z
            v = A[j:, j] - L[j:, 0:j] @ z

        if j < m - 1:
            L[j+1:m, j] = v[1:] / v[0]

        U[j, j] = v[0]

    return L, U


A = np.array([
    [2,  1,  0,  0],
    [4,  5,  1,  0],
    [6, 13,  8,  1],
    [3,  4,  5,  6],
    [7,  2,  9,  1],
    [8,  1,  2,  3]
], dtype=float)

L, U = lu_mn(A)
print("L = \n", L)
print("U = \n", U)
print(A - L @ U)

