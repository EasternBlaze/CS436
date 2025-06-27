import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo


iris = fetch_ucirepo(id=53)  # Iris dataset
x = iris.data.features  # pandas DataFrame
y = iris.data.targets   # pandas DataFrame



# Feature Filtering

data = pd.concat([x, y], axis=1) # Merge features and targets
data.columns = list(x.columns) + ['class'] # Rename label column
# Filter for 'Iris-setosa' and 'Iris-versicolor'
data = data[data['class'].isin(['Iris-setosa', 'Iris-versicolor'])]

x = data[['sepal length', 'sepal width']].to_numpy() # Feature selection

def label_flower(name): # Map class labels: versicolor = 0, setosa = 1
    if name == 'Iris-versicolor': return 0
    elif name == 'Iris-setosa': return 1
    else: return -1 # Virginicia isn’t being used
binary_labels = data['class'].apply(label_flower).to_numpy()



# Plot for features

# Create class-specific masks for readability
versi = np.where(binary_labels == 0)[0]
seto = np.where(binary_labels == 1)[0]

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(x[versi, 0], x[versi, 1],
            color='blue', marker='s', label='Versicolor (0)')
plt.scatter(x[seto, 0], x[seto, 1],
            color='red', marker='o', label='Setosa (1)')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Feature Distribution: Sepal Length vs Sepal Width')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()



# Data Splitting

# Combine everything into one DataFrame
data['label'] = binary_labels
x_full = data[['sepal length', 'sepal width']]
y_full = data['label']

# Shuffle the DataFrame with fixed seed
shuffled = data.sample(frac=1, random_state=31).reset_index(drop=True)

# Split into train and test sets
split_point = int(0.8 * len(shuffled))
train = shuffled.iloc[:split_point]
test = shuffled.iloc[split_point:]

# Extract x and y from train/test sets
x_train = train[['sepal length', 'sepal width']].to_numpy()
y_train = train['label'].to_numpy()
x_test = test[['sepal length', 'sepal width']].to_numpy()
y_test = test['label'].to_numpy()

# Add bias (intercept) column
x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])



# Logistic Regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights: [bias, w1, w2]
w = np.zeros(x_train.shape[1])
learning_rate = 0.05
epochs = 200
train_acc = []

for epoch in range(epochs):
    z = np.dot(x_train, w) # Calculate log odds
    pred = sigmoid(z) # Convert to probability
    error = pred - y_train
    gradTemp = error * pred * (1 - pred) # line before 'final update'
    grad = np.dot(x_train.T, gradTemp) / len(y_train)
    w = w - (learning_rate * grad)

    pred_label = (pred >= 0.5).astype(int) 
    accu = np.mean(pred_label == y_train)
    train_acc.append(accu)


    if (epoch + 1) % 10 == 0:
        loss = np.mean((pred - y_train) ** 2)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accu:.4f}, weight={w}")

print("\nFinal Model Weights:", w)


# Plot for Learning Curve 
plt.figure(figsize=(8,6))
plt.plot(range(epochs), train_acc, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve (Training Accuracy vs Epochs)')
plt.grid(True)
plt.legend()
plt.show()


# Final Accuracy
finPred = sigmoid(np.dot(x_test, w))
finClass = (finPred >= 0.5).astype(int)
finAcc = np.mean(finClass == y_test)
print(f"\nFinal Accuracy: {finAcc*100:.2f}%")
