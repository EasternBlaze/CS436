import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

# 1. Prepare the Data [5 pts]
RANDOM_STATE = 200
N_SAMPLES = 3000
N_CLUSTERS = 40

X, y_true = make_blobs(
    n_samples=N_SAMPLES,
    centers=N_CLUSTERS,
    cluster_std=0.8,
    random_state=RANDOM_STATE
)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true, test_size=0.2, random_state=RANDOM_STATE
)

class knn:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    # euclidean distance
    def eucli(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # store dataset
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # predict for single data
    def predictSingle(self, x_test_point):
        distances = [self.eucli(x_test_point, x_train_point) for x_train_point in self.X_train]

        # indices
        k_nearest_indices = np.argsort(distances)[:self.k]

        # labels
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # Predict the label by majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    # predict for set
    def predict(self, X_test):
        predictions = [self.predictSingle(x) for x in X_test]
        return np.array(predictions)

# accuracy calculation
def calcAcc(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = {}

for k in k_values:
    knn_classifier = knn(k=k)
    knn_classifier.fit(X_train, y_train)
    predictions = knn_classifier.predict(X_test)
    accuracy = calcAcc(y_test, predictions)
    accuracies[k] = accuracy
    print(f"k = {k}: {accuracy:.4f}")

# Finding best k
bestK = max(accuracies, key=accuracies.get)
bestAcc = accuracies[bestK]

print(f"k = {bestK},  accuracy = {bestAcc:.4f}")
