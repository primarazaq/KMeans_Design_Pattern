from k_means import K_Means

class K_MeansBuilder:
    def __init__(self):
        self.k = 3
        self.tolerance = 0.0001
        self.max_iterations = 10

    def set_k(self, k):
        self.k = k
        return self

    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
        return self

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations
        return self

    def build(self):
        return K_Means(self.k, self.tolerance, self.max_iterations)
