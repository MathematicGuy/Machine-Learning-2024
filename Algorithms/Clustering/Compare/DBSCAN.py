import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

# Generate datasets
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

# Set up DBSCAN parameters
params = {
    "eps": 0.3,  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
    "min_samples": 7,  # The number of samples in a neighborhood for a point to be considered as a core point
}

# List of datasets and their specific parameters
datasets = [
    (noisy_circles, {}),
    (noisy_moons, {}),
    (varied, {"eps": 0.18}),
    (aniso, {"eps": 0.15}),
    (blobs, {}),
    (no_structure, {}),
]

# Extract the noisy_circles dataset-pcb from the datasets list
X, y = datasets[0][0]

# Normalize dataset-pcb for easier parameter selection
X = StandardScaler().fit_transform(X)

# Apply DBSCAN
dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
t0 = time.time()
dbscan.fit(X)
t1 = time.time()

# Plot result
plt.figure(figsize=(10, 10))
plt.title("Noisy Circles Dataset")

labels = dbscan.labels_

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & (labels == k)]
    plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6)

plt.show()