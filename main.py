import math
from utils.io import read_input_2d, read_input_3d
import sys



def matrix(n, m=None, default=0):
    m = m or n
    return [[default] * m for _ in range(n)]


def dynamic_time_warping(s, t):
    n, m = len(s), len(t)
    dtw = matrix(n, m)

    for i in range(n):
        dtw[i][0] = math.inf
    for i in range(m):
        dtw[0][i] = math.inf
    dtw[0][0] = 0

    for i in range(n):
        for j in range(m):
            dist = abs(s[i] - t[j])
            dtw[i][j] = dist + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
            sys.stdout.write(str(dtw[i][j]) + " ")
        print('')

    return dtw[n-1][m-1]


train, test, labels = read_input_3d()
dynamic_time_warping([1,2,3,4,5,6],[1,2,3,4,5,6])


