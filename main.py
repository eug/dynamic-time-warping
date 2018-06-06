import math
from utils.io import read_input_2d, read_input_3d
import sys
from tqdm import tqdm


def matrix(n, m=None, default=0):
    m = m or n
    return [[default] * m for _ in range(n)]


def dtw_naive(s, t, prune_score=math.inf):
    n, m = len(s), len(t)
    dtw = matrix(n, m, default=math.inf)
    dtw[0][0] = 0
    min_row_score = -1

    for i in range(1, n):
        min_row_score = math.inf

        for j in range(1, m):
            dtw[i][j] = abs(s[i] - t[j]) + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
            # min_row_score = min(min_row_score, dtw[i][j])

        if min_row_score > prune_score:
            return min_row_score
    
    return min_row_score


def dtw_sakoe_chiba(s, t, w, prune_score=math.inf):
    n, m = len(s), len(t)
    w = max(w, abs(n - m) + 1)
    dtw = matrix(n, m, default=math.inf)
    dtw[0][0] = 0
    min_row_score = -1

    for i in range(1, n):
        min_row_score = math.inf

        for j in range(max(1, i-w), min(m, i+w)):
            dtw[i][j] = abs(s[i] - t[j]) + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
            min_row_score = min(min_row_score, dtw[i][j])

        if min_row_score > prune_score:
            return min_row_score

    return min_row_score


def predict(train, test):
    y_pred = []
    for tsl, t in tqdm(test):
        min_label, min_score = -1, math.inf
        for trl, s in train:
            # score = dtw_sakoe_chiba(s, t, 5, min_score)
            score = dtw_naive(s, t, min_score)
            if score < min_score:
                min_score = score
                min_label = trl
        y_pred.append(tsl == min_label)
    
    return y_pred


train, test, labels = read_input_2d()
y_pred = predict(train, test)
print(sum(y_pred))
print(len(y_pred))
print( sum(y_pred) / len(y_pred) * 100 )

# print(len(train[0][1]), len(test[0][1]))
# n, m = len(train[0][1]), len(test[0][1])
# dtw = dtw_naive(train[0][1], test[0][1])
# dtw = dtw_sakoe_chiba(train[0][1], test[0][1], 5)
# print( dtw[len(train[0][1])-1][len(test[0][1])-1] )
# print(dtw)
# for i in range(n):
#     for j in range(m):
#         sys.stdout.write(str(dtw[i][j]) + "\t")
#     sys.stdout.write("\n")