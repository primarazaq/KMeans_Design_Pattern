import matplotlib.pyplot as plt

class KMeansPlotter:
    def plot(self, km):
        colors = 10 * ["r", "c", "k", "g", "b"]

        arr = []
        for centroid in km.centroids:
            plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")

        for classification in km.classes:
            color = colors[classification]
            for features in km.classes[classification]:
                plt.scatter(features[0], features[1], color=color, s=30)
                arr.append([classification])

        print("Total Data : ", len(arr), "data")
        for i in range(km.k):
            print("Kluster ", i, ": ", arr.count([i]), "data")
        plt.show()
