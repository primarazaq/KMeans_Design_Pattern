
from k_means import K_Means

class KMeansPool:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=10):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.pool = []

    def acquire(self):
        if len(self.pool) > 0:
            return self.pool.pop()
        else:
            return K_Means(self.k, self.tolerance, self.max_iterations)

    def release(self, k_means):
        self.pool.append(k_means)
