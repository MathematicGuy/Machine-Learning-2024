import os
import sklearn.metrics
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

"""
    data = [
        ['Age', 'Income($)'],
        ['27', '70000'], ['29', '90000'], ['29', '61000'], ['28', '60000'],
        ['42', '150000'], ['39', '155000'], ['41', '160000'], ['38', '162000'],
        ['36', '156000'], ['35', '130000'], ['37', '137000'], ['26', '45000'],
        ['27', '48000'], ['28', '51000'], ['29', '49500'], ['32', '53000'],
        ['40', '65000'], ['41', '63000'], ['43', '64000'], ['39', '80000'],
        ['41', '82000'], ['39', '58000']
    ]
"""

os.environ['OMP_NUM_THREADS'] = '1'
data = []
# Open the CSV file in read mode
with open('Income.csv', 'r') as cvf:
    # Create a reader object
    reader = csv.reader(cvf)

    # Iterate through the rows in the CSV file
    for row in reader:
        # Access reach element in the row
        data.append(row[:][-2:])
    # print("data =", data)
    # print()

# Your data

# Convert to DataFrame
df = pd.DataFrame(data[1:], columns=data[0])  # data[:1] mean choose every row except for the 0th row
# Convert columns to numeric
df['Age'] = pd.to_numeric(df['Age'])
df['Income($)'] = pd.to_numeric(df['Income($)'])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

df_scaled = scaler.fit_transform(df)
scaled_data_list = np.array(df_scaled)  # convert to numpy array for ML
print('dataset-pcb:', df_scaled)


class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=200):
        # ? Make sure the centroid generated within the given Axis
        # ? amin/amax - axis-min/axis-max
        # ? K: how many centroid we want to have
        print("shape:", X.shape)
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iterations):
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

            if np.max(self.centroids - np.array(cluster_centers)) < 0.001:
                break
            else:
                self.centroids = np.array(cluster_centers)

            return y


random_points = scaled_data_list
kmeans = KMeansClustering(k=3)  # Assigned k=3 to KMeansClustering Object
labels = kmeans.fit(random_points)

print("Clusters Label:", labels)
# ari = adjusted_rand_score(labels, labels)
# print(ari)

plt.figure(figsize=(10, 6))
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)  # x, y, color
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
            marker="*", s=200)
plt.title('K-Means Clustering Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
