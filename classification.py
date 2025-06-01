import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import issparse

# Load preprocessed data
X_train = np.load('X_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

# Inspect data
print("Type of X_train:", type(X_train))
print("Shape of X_train:", X_train.shape if isinstance(X_train, np.ndarray) else "Not an array")
print("Type of y_train:", type(y_train))
print("Shape of y_train:", y_train.shape if isinstance(y_train, np.ndarray) else "Not an array")

# Handle 0D arrays
if isinstance(X_train, np.ndarray) and X_train.shape == ():
    X_train = X_train.item()
if isinstance(X_test, np.ndarray) and X_test.shape == ():
    X_test = X_test.item()

# Convert sparse matrices to dense
if issparse(X_train):
    X_train = X_train.toarray()
if issparse(X_test):
    X_test = X_test.toarray()

# Implement KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

# Implement Naïve Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)

# Implement Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Save predictions
np.save('knn_predictions.npy', knn_preds)
np.save('nb_predictions.npy', nb_preds)
np.save('rf_predictions.npy', rf_preds)

# Output verification
print("Sample KNN predictions:", knn_preds[:5])
print("Sample Naïve Bayes predictions:", nb_preds[:5])
print("Sample Random Forest predictions:", rf_preds[:5])
print("KNN Prediction distribution:", np.unique(knn_preds, return_counts=True))
print("Naïve Bayes Prediction distribution:", np.unique(nb_preds, return_counts=True))
print("Random Forest Prediction distribution:", np.unique(rf_preds, return_counts=True))
print("True label distribution:", np.unique(y_test, return_counts=True))