import numpy as np
import pandas as pd


# np.random.seed(2)
def det(A):  # 求A的行列式
    if len(A) <= 0:
        return None
    elif len(A) == 1:
        return A[0][0]
    else:
        s = 0
        for i in range(len(A)):
            n = [[row[a] for a in range(len(A)) if a != i] for row in A[1:]]  # 这里生成余子式
            s += A[0][i] * det(n) * (-1) ** (i)
        return s


def Master_Sequential(A, n):  # 判断A的k阶顺序主子式是否非零，非零满足LU分解的条件
    for i in range(0, n):
        Master = np.zeros([i + 1, i + 1])
        for row in range(0, i + 1):
            for a in range(0, i + 1):
                Master[row][a] = A[row][a]
        if det(Master) == 0:
            done = False
            return done


def LU_decomposition(A):
    n = len(A[0])
    L = np.zeros([n, n])
    U = np.zeros([n, n])
    for i in range(n):
        L[i][i] = 1
        if i == 0:
            U[0][0] = A[0][0]
            for j in range(1, n):
                U[0][j] = A[0][j]
                L[j][0] = A[j][0] / U[0][0]
        else:
            for j in range(i, n):  # U
                temp = 0
                for k in range(0, i):
                    temp = temp + L[i][k] * U[k][j]
                U[i][j] = A[i][j] - temp
            for j in range(i + 1, n):  # L
                temp = 0
                for k in range(0, i):
                    temp = temp + L[j][k] * U[k][i]
                L[j][i] = (A[j][i] - temp) / U[i][i]
    return L, U


if __name__ == '__main__':
    n = 3
    # A=[[2,1,-5, 1], [1, -3, 0, -6], [0, 2,-1, 2], [1, 4, -7, 6]]
    # A = np.random.randint(1, 10, size=[n, n])
    A = [[2, 1, -5, 1],
         [1, -3, 0, -6],
         [0, 2, -1, 2],
         [1, 4, -7, 6]]

    print('A矩阵：\n', A)
    if Master_Sequential(A, n) != False:
        L, U = LU_decomposition(A)
        print('L矩阵：\n', L, '\nU矩阵：\n', U)
    else:
        print('A的k阶主子式不全非零，不满足LU分解条件。')