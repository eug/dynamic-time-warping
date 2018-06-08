

def abs_distance(p, q):
    return sum(abs(a - b) for a, b in zip(p, q))

def euclidean_distance(p, q):
    return sum((a - b)**2 for a, b in zip(p, q))**0.5