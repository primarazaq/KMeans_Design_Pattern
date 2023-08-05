from kmeans import KMeans

class KMeansFactory:
    def create_algorithm(self, k, tolerance, max_iterations):
        return KMeans(k, tolerance, max_iterations)
