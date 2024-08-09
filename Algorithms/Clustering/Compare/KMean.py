import numpy as np


class KMeansClustering:
    def __init__(self, k=3, max_iter=200):
        self.k = k
        self.centroids = None
        self.max_iter = max_iter

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iter):
        #? Make sure the centroid generated within the given Axis
        #? amin/amax - axis-min/axis-max
        #? K: how many centroids we want to have
        print("shape:", X.shape)
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iter):
            y = []
            for data_points in X:
                distances = KMeansClustering.euclidean_distance(data_points, self.centroids)
                # ? argmin(distances): find the index of the smallest distance in distances array
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            y = np.array(y)
            # print(y)

            # Re-Adjust the Centroid Position. Base on these label
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))  # append cluster indexes
            # print('cluster_indices:', cluster_indices)

            cluster_centers = []  # reposition the centroid

            for i, indices in enumerate(cluster_indices):
                # for when there 100th centroid and only 3 clusters. Some of the Centroids will have empty cluster indices.
                if len(indices) == 0:
                    # set the empty centroid as the new centroid
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            # print('cluster_centers:', cluster_centers)

            if np.max(self.centroids - np.array(cluster_centers)) < 0.01:
                break
            else:
                self.centroids = np.array(cluster_centers)

            return y
