import math

def matrix(n, m=None, default=0):
    m = m or n
    return [[default] * m for _ in range(n)]

def dtw_naive(s, t, dist, prune_score=None):
    min_row_score = -1
    n, m = len(s), len(t)
    dtw = matrix(n, m)

    for i in range(n):
        min_row_score = math.inf
        for j in range(m):

            min_prev_cost = 0
            if i > 0 and j > 0:
                min_prev_cost = min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
            elif i > 0 and j == 0:
                min_prev_cost = dtw[i-1][j]
            elif i == 0 and j > 0:
                min_prev_cost = dtw[i][j-1]
            
            cost = dist(s[i], t[j])
            dtw[i][j] = cost + min_prev_cost
            min_row_score = min(min_row_score, dtw[i][j])

        if prune_score and min_row_score > prune_score:
            return min_row_score

    return min_row_score


def dtw_sakoe_chiba(s, t, w, dist, prune_score=None):
    min_row_score = -1
    n, m = len(s), len(t)
    w = max(w, abs(n - m) + 1)
    dtw = matrix(n, m, default=math.inf)

    for i in range(n):
        min_row_score = math.inf
        for j in range(max(0, i-w), min(m, i+w)):

            min_prev_cost = 0
            if i != 0 or j != 0:
                min_prev_cost = min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

            cost = dist(s[i], t[j])
            dtw[i][j] = cost + min_prev_cost
            min_row_score = min(min_row_score, dtw[i][j])

        if prune_score and min_row_score > prune_score:
            return min_row_score

    return min_row_score