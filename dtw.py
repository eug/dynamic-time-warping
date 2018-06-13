import math

def matrix(n, m=None, default=0):
    m = m or n
    return [[default] * m for _ in range(n)]

def dtw_naive(s, t, dist, prune_score=None):
    n, m = len(s), len(t)
    dtw = matrix(n, m, default=math.inf)
    dtw[0][0] = 0

    for i in range(1, n):
        for j in range(1, m):
            cost = dist(s[i], t[j])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

        if prune_score and min(dtw[i]) > prune_score:
            return math.inf

    return dtw[-1][-1]

def dtw_sakoe_chiba(s, t, w, dist, prune_score=None):
    n, m = len(s), len(t)
    w = max(w, abs(n - m) + 1)
    dtw = matrix(n, m, default=math.inf)
    dtw[0][0] = 0

    for i in range(1, n):
        for j in range(max(1, i-w), min(m, i+w)):
            cost = dist(s[i], t[j])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

        if prune_score and min(dtw[i]) > prune_score:
            return math.inf

    return dtw[-1][-1]
