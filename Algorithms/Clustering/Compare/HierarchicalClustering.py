import numpy as np

class HierarchicalClustering:
    def __init__(self, k=3):
        # Initialize the number of clusters and labels
        self.k = k
        self.labels_ = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        # Calculate the Euclidean distance between a data point and centroids
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X):
        n_samples = X.shape[0]
        # Initialize clusters with each data point as its own cluster
        clusters = {i: [i] for i in range(n_samples)}
        # Initialize distances with infinity
        distances = np.full((n_samples, n_samples), np.inf)

        # Calculate initial pairwise distances between all data points
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = distances[j, i] = np.linalg.norm(X[i] - X[j])

        # Merge clusters until the desired number of clusters is reached
        while len(clusters) > self.k:
            min_dist = np.inf
            to_merge = None

            # Find the pair of clusters with the minimum distance
            for i in clusters:
                for j in clusters:
                    if i != j:
                        dist = np.mean([distances[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                        if dist < min_dist:
                            min_dist = dist
                            to_merge = (i, j)

            # Merge the closest clusters
            i, j = to_merge
            clusters[i].extend(clusters[j])
            del clusters[j]

            # Update distances after merging
            for m in clusters:
                if m != i:
                    distances[i, m] = distances[m, i] = np.mean(
                        [distances[p1, p2] for p1 in clusters[i] for p2 in clusters[m]])

        # Assign labels to data points based on the final clusters
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, points in clusters.items():
            for point in points:
                self.labels_[point] = cluster_id

        return self.labels_

# Example usage:
# X, y = make_blobs(n_samples=500, centers=3, cluster_std=[0.5, 2.2, 2.2], random_state=42)
# hc = HierarchicalClustering(k=3)
# labels = hc.fit(X)