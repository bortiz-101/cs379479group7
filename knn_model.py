import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# code converted from HW assignment 
# To-Do: limit overhead, try new alg bc original KNN too memory and time intensive

load_dotenv()

cwd = os.getenv('CWD')
data_path = os.getenv('DATA_PATH')
images_path = os.getenv('IMAGES_PATH')
train_hdf5 = os.getenv('TRAIN_HDF5')
train_csv = os.getenv('TRAIN_CSV')

train_data = pd.read_csv(os.path.join(data_path, 'train-metadata.csv'))
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
train_data.fillna(train_data.mode().iloc[0], inplace=True)
train_data = pd.get_dummies(train_data, columns=['sex', 'anatom_site_general'], drop_first=True)

#define features and labels
X = train_data.drop(columns=['target'])
y = train_data['target']

#split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#scale and transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
X_val_scaled = scaler.transform(X_val.select_dtypes(include=[np.number]))
X_train_scaled = np.array(X_train_scaled)
X_val_scaled = np.array(X_val_scaled)
y_train = np.array(y_train)
y_val = np.array(y_val)


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


k_values = range(1, 6) # range, not practical- too much time
accuracies = []

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train_scaled, y_train)
    predictions = knn.predict(X_val_scaled)
    accuracy = np.mean(predictions == y_val)
    accuracies.append(accuracy)
    auc = roc_auc_score(y_val, predictions)

    print(f"Accuracy for k={k}: {accuracy:.3f}, AUC: {auc:.3f}")

best_k = k_values[np.argmax(accuracies)]
print(f"Best k value: {best_k}")

knn = KNN(k=best_k)
knn.fit(X_train_scaled, y_train)
final_predictions = knn.predict(X_val_scaled)

final_auc = roc_auc_score(y_val, final_predictions)
conf_matrix = confusion_matrix(y_val, final_predictions)
classification_rep = classification_report(y_val, final_predictions)

print(f"Final AUC: {final_auc:.3f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
