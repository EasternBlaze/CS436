import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate consistent 2D synthetic data using make_blobs from sklearn.datasets. [5pts]
RANDOM_STATE = 42
N_SAMPLES = 300
N_CLUSTERS = 4

X, y_true = make_blobs(
    n_samples=N_SAMPLES,
    centers=N_CLUSTERS,
    cluster_std=0.8,
    random_state=RANDOM_STATE
)

# 2. Implement the K-Means Algorithm from Scratch [20 pts]

class km:
    def __init__(self, k=3, max_iters=255, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.iterations = 0

    def eucli(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def fit(self, X):

        n_samples, n_features = X.shape

        # initialize centroids ramdomly
        np.random.seed(self.random_state)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        # Update centroids until convergence (or set a maximum number of iterations)
        for i in range(self.max_iters):
            self.iterations = i + 1
            
            # assign to the closest centroid
            self.labels = np.zeros(n_samples)
            for idx, point in enumerate(X):
                distances = [self.eucli(point, c) for c in self.centroids]
                self.labels[idx] = np.argmin(distances)

            # Store old centroids to check for convergence
            old_centroids = np.copy(self.centroids)

            # Calculate new centroids as the mean of assigned points
            for cluster_idx in range(self.k):
                points_in_cluster = X[self.labels == cluster_idx]
                if len(points_in_cluster) > 0:
                    self.centroids[cluster_idx] = np.mean(points_in_cluster, axis=0)
            
            # If centroids no longer change, finish
            if np.allclose(old_centroids, self.centroids):
                break
        
        return self

kmeans = km(k=N_CLUSTERS, random_state=RANDOM_STATE)
kmeans.fit(X)
final_labels = kmeans.labels
final_centroids = kmeans.centroids

# Visualization
plt.figure(figsize=(12, 8))
# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', marker='o', alpha=0.8, label='Data Points')

# final centroids
plt.scatter(
    final_centroids[:, 0], 
    final_centroids[:, 1], 
    c='red',              
    s=250,                  
    marker='.',              
    linewidth=1.5, 
    label='Final Centroids'
)

plt.title('K-Means Clustering from Scratch (k=4)', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(f"The algorithm converged in **{kmeans.iterations} iterations**.")