import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
    [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]
])

# center the dataset
mean_vec = np.mean(data, axis=0)
centered_data = data - mean_vec

print("\n centered dataset")
print(np.round(centered_data, 4))

# 2x2 covariance matrix of the centered dataset
cov_matrix = np.cov(centered_data.T)
print("\n2x2 Covariance Matrix")
print(np.round(cov_matrix, 4))


# [5pts] Perform eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"\nEigenvalues: {np.round(eigenvalues, 4)}")
print(f"Eigenvectors:\n{np.round(eigenvectors, 4)}")


# [5pts] Identify the first principal component
pc1_index = np.argmax(eigenvalues)
pc1_eigenvector = eigenvectors[:, pc1_index]
print(f"\nFirst Principal Component: {np.round(pc1_eigenvector, 4)}")


# [5pts] Project the centered data onto the first principal component
proj1D = centered_data @ pc1_eigenvector
print("\nProjected 1D Data")
print(np.round(proj1D, 4))


# visualization
plt.figure(figsize=(9, 7))

# original data points
plt.scatter(data[:, 0], data[:, 1], alpha=0.9, label='Original Data Points')

# arrow from mean of the data, scaled by eigenvalue
scale_factor = 3 * np.sqrt(eigenvalues[pc1_index])
arrow_origin = mean_vec
arrow_vector = pc1_eigenvector * scale_factor

plt.arrow(
    arrow_origin[0], arrow_origin[1],
    arrow_vector[0], arrow_vector[1],
    head_width=0.1, head_length=0.2, fc='r', ec='r',
    label='First Principal Component'
)

plt.title('Principal Component Analysis from Scratch')
plt.xlabel('Feature x')
plt.ylabel('Feature y')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.show()