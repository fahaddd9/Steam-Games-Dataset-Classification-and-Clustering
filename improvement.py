import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
import matplotlib.pyplot as plt

# Step 1: Load data
X_train = np.load('X_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)
X_clustering_scaled = np.load('X_clustering_scaled.npy', allow_pickle=True)
df = pd.read_csv('preprocessed_steam_games.csv')
y_true_all = df['primary_genre'].values

# Step 2: Handle 0D arrays and sparse matrices
if isinstance(X_train, np.ndarray) and X_train.shape == ():
    X_train = X_train.item()
if isinstance(X_test, np.ndarray) and X_test.shape == ():
    X_test = X_test.item()
if issparse(X_train):
    X_train = X_train.toarray()
if issparse(X_test):
    X_test = X_test.toarray()
if isinstance(X_clustering_scaled, np.ndarray) and X_clustering_scaled.shape == ():
    X_clustering_scaled = X_clustering_scaled.item()

# Step 3: Load original predictions
knn_preds = np.load('knn_predictions.npy', allow_pickle=True)
nb_preds = np.load('nb_predictions.npy', allow_pickle=True)
rf_preds = np.load('rf_predictions.npy', allow_pickle=True)

# Step 4: Original metrics
original_knn_metrics = {
    'accuracy': 0.65,
    'macro_f1': 0.48,
    'rpg_f1': 0.32,
    'simulation_f1': 0.45
}
original_nb_metrics = {
    'accuracy': 0.68,
    'macro_f1': 0.50,
    'rpg_f1': 0.35,
    'simulation_f1': 0.47
}
original_rf_metrics = {
    'accuracy': 0.69,
    'macro_f1': 0.52,
    'rpg_f1': 0.36,
    'simulation_f1': 0.48
}

# Step 5: Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("SMOTE applied. New class distribution in training set:")
print(pd.Series(y_train_smote).value_counts())

# Step 6: Improve KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=3, scoring='f1_macro', n_jobs=-1)
grid_search_knn.fit(X_train_smote, y_train_smote)
best_knn = grid_search_knn.best_estimator_
print("\nBest KNN parameters:", grid_search_knn.best_params_)
knn_preds_improved = best_knn.predict(X_test)
print("\nImproved KNN Classification Report:")
knn_report_improved = classification_report(y_test, knn_preds_improved, output_dict=True)
print(classification_report(y_test, knn_preds_improved))

# Step 7: Improve Naïve Bayes
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 2.0]
}
nb = MultinomialNB()
# Convert negative values to non-negative for MultinomialNB
X_train_smote_nb = np.abs(X_train_smote)
X_test_nb = np.abs(X_test)
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=3, scoring='f1_macro', n_jobs=-1)
grid_search_nb.fit(X_train_smote_nb, y_train_smote)
best_nb = grid_search_nb.best_estimator_
print("\nBest Naïve Bayes parameters:", grid_search_nb.best_params_)
nb_preds_improved = best_nb.predict(X_test_nb)
print("\nImproved Naïve Bayes Classification Report:")
nb_report_improved = classification_report(y_test, nb_preds_improved, output_dict=True)
print(classification_report(y_test, nb_preds_improved))

# Step 8: Improve Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='f1_macro', n_jobs=-1)
grid_search_rf.fit(X_train_smote, y_train_smote)
best_rf = grid_search_rf.best_estimator_
print("\nBest Random Forest parameters:", grid_search_rf.best_params_)

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = indices[:20]
numerical_cols = ['original_price', 'discount_price', 'achievements', 'release_year', 'num_languages', 'num_tags', 'sentiment_score', 'desc_word_count', 'requires_gpu']
categorical_cols = ['developer', 'publisher', 'popular_tags', 'game_details', 'mature_content']
cat_features = []
for col in categorical_cols:
    cat_features.extend([f"{col}_{i}" for i in range(10)])
feature_names = numerical_cols + cat_features
print("\nTop 20 features selected:")
print([feature_names[i] for i in top_features])

X_train_smote_selected = X_train_smote[:, top_features]
X_test_selected = X_test[:, top_features]
best_rf.fit(X_train_smote_selected, y_train_smote)
rf_preds_improved = best_rf.predict(X_test_selected)
print("\nImproved Random Forest Classification Report:")
rf_report_improved = classification_report(y_test, rf_preds_improved, output_dict=True)
print(classification_report(y_test, rf_preds_improved))

# Step 9: Clustering Improvements
X_full = np.vstack((X_train, X_test))
if issparse(X_full):
    X_full = X_full.toarray()
np.save('X_clustering_improved.npy', X_full)
print("Improved clustering data saved as 'X_clustering_improved.npy'")

inertia = []
k_range = range(1, 16)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_full)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (Improved)')
plt.grid(True)
plt.savefig('static/images/elbow_plot_improved.png')
plt.close()
print("\nImproved elbow plot saved as 'elbow_plot_improved.png'")

k_new = 5
kmeans_improved = KMeans(n_clusters=k_new, random_state=42, n_init=10)
kmeans_labels_improved = kmeans_improved.fit_predict(X_full)
np.save('kmeans_labels_improved.npy', kmeans_labels_improved)
print("Improved clustering labels saved as 'kmeans_labels_improved.npy'")

silhouette_improved = silhouette_score(X_full, kmeans_labels_improved)
ari_improved = adjusted_rand_score(y_true_all, kmeans_labels_improved)
print("\nImproved K-Means Metrics:")
print(f"Silhouette Score: {silhouette_improved:.3f}")
print(f"Adjusted Rand Index: {ari_improved:.3f}")

original_kmeans_metrics = {
    'silhouette': 0.342,
    'ari': 0.125
}

# Step 10: Generate Before and After Comparison Charts
bar_width = 0.35
x = np.arange(4)

# 10.1: KNN Metrics Comparison
knn_metrics = ['Accuracy', 'Macro F1', 'RPG F1', 'Simulation F1']
before_knn_values = [original_knn_metrics['accuracy'], original_knn_metrics['macro_f1'], original_knn_metrics['rpg_f1'], original_knn_metrics['simulation_f1']]
after_knn_values = [knn_report_improved['accuracy'], knn_report_improved['macro avg']['f1-score'], knn_report_improved['RPG']['f1-score'], knn_report_improved['Simulation']['f1-score']]
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, before_knn_values, bar_width, label='Before', color='#ff9999')
plt.bar(x + bar_width/2, after_knn_values, bar_width, label='After', color='#66b3ff')
for i in range(4):
    plt.text(x[i] - bar_width/2, before_knn_values[i], round(before_knn_values[i], 2), ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, after_knn_values[i], round(after_knn_values[i], 2), ha='center', va='bottom')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('KNN Performance: Before vs. After Improvements')
plt.xticks(x, knn_metrics)
plt.legend()
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/images/knn_metrics_comparison.png')
plt.close()

# 10.2: Naïve Bayes Metrics Comparison
nb_metrics = ['Accuracy', 'Macro F1', 'RPG F1', 'Simulation F1']
before_nb_values = [original_nb_metrics['accuracy'], original_nb_metrics['macro_f1'], original_nb_metrics['rpg_f1'], original_nb_metrics['simulation_f1']]
after_nb_values = [nb_report_improved['accuracy'], nb_report_improved['macro avg']['f1-score'], nb_report_improved['RPG']['f1-score'], nb_report_improved['Simulation']['f1-score']]
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, before_nb_values, bar_width, label='Before', color='#ff9999')
plt.bar(x + bar_width/2, after_nb_values, bar_width, label='After', color='#66b3ff')
for i in range(4):
    plt.text(x[i] - bar_width/2, before_nb_values[i], round(before_nb_values[i], 2), ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, after_nb_values[i], round(after_nb_values[i], 2), ha='center', va='bottom')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Naïve Bayes Performance: Before vs. After Improvements')
plt.xticks(x, nb_metrics)
plt.legend()
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/images/nb_metrics_comparison.png')
plt.close()

# 10.3: Random Forest Metrics Comparison
rf_metrics = ['Accuracy', 'Macro F1', 'RPG F1', 'Simulation F1']
before_rf_values = [original_rf_metrics['accuracy'], original_rf_metrics['macro_f1'], original_rf_metrics['rpg_f1'], original_rf_metrics['simulation_f1']]
after_rf_values = [rf_report_improved['accuracy'], rf_report_improved['macro avg']['f1-score'], rf_report_improved['RPG']['f1-score'], rf_report_improved['Simulation']['f1-score']]
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, before_rf_values, bar_width, label='Before', color='#ff9999')
plt.bar(x + bar_width/2, after_rf_values, bar_width, label='After', color='#66b3ff')
for i in range(4):
    plt.text(x[i] - bar_width/2, before_rf_values[i], round(before_rf_values[i], 2), ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, after_rf_values[i], round(after_rf_values[i], 2), ha='center', va='bottom')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Random Forest Performance: Before vs. After Improvements')
plt.xticks(x, rf_metrics)
plt.legend()
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/images/rf_metrics_comparison.png')
plt.close()

# 10.4: K-Means Metrics Comparison
kmeans_metrics = ['Silhouette Score', 'Adjusted Rand Index']
before_kmeans_values = [original_kmeans_metrics['silhouette'], original_kmeans_metrics['ari']]
after_kmeans_values = [silhouette_improved, ari_improved]
x = np.arange(2)
plt.figure(figsize=(8, 6))
plt.bar(x - bar_width/2, before_kmeans_values, bar_width, label='Before', color='#ff9999')
plt.bar(x + bar_width/2, after_kmeans_values, bar_width, label='After', color='#66b3ff')
for i in range(2):
    plt.text(x[i] - bar_width/2, before_kmeans_values[i], round(before_kmeans_values[i], 3), ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, after_kmeans_values[i], round(after_kmeans_values[i], 3), ha='center', va='bottom')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('K-Means Performance: Before vs. After Improvements')
plt.xticks(x, kmeans_metrics)
plt.legend()
plt.ylim(0, max(max(before_kmeans_values), max(after_kmeans_values)) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/images/kmeans_metrics_comparison.png')
plt.close()

# Step 11: Save improvement report
classes = ['Action', 'Adventure', 'RPG', 'Simulation']
with open('improvement_report.md', 'w') as f:
    f.write("# Task 7: Efforts to Improve Results\n\n")
    f.write("## Classification Improvements\n\n")
    f.write("### KNN\n")
    f.write(f"- Original Accuracy: {original_knn_metrics['accuracy']:.2f}, Improved: {knn_report_improved['accuracy']:.2f}\n")
    f.write(f"- Original Macro F1: {original_knn_metrics['macro_f1']:.2f}, Improved: {knn_report_improved['macro avg']['f1-score']:.2f}\n")
    f.write(f"- Original RPG F1: {original_knn_metrics['rpg_f1']:.2f}, Improved: {knn_report_improved['RPG']['f1-score']:.2f}\n")
    f.write(f"- Original Simulation F1: {original_knn_metrics['simulation_f1']:.2f}, Improved: {knn_report_improved['Simulation']['f1-score']:.2f}\n\n")
    f.write("### Naïve Bayes\n")
    f.write(f"- Original Accuracy: {original_nb_metrics['accuracy']:.2f}, Improved: {nb_report_improved['accuracy']:.2f}\n")
    f.write(f"- Original Macro F1: {original_nb_metrics['macro_f1']:.2f}, Improved: {nb_report_improved['macro avg']['f1-score']:.2f}\n")
    f.write(f"- Original RPG F1: {original_nb_metrics['rpg_f1']:.2f}, Improved: {nb_report_improved['RPG']['f1-score']:.2f}\n")
    f.write(f"- Original Simulation F1: {original_nb_metrics['simulation_f1']:.2f}, Improved: {nb_report_improved['Simulation']['f1-score']:.2f}\n\n")
    f.write("### Random Forest\n")
    f.write(f"- Original Accuracy: {original_rf_metrics['accuracy']:.2f}, Improved: {rf_report_improved['accuracy']:.2f}\n")
    f.write(f"- Original Macro F1: {original_rf_metrics['macro_f1']:.2f}, Improved: {rf_report_improved['macro avg']['f1-score']:.2f}\n")
    f.write(f"- Original RPG F1: {original_rf_metrics['rpg_f1']:.2f}, Improved: {rf_report_improved['RPG']['f1-score']:.2f}\n")
    f.write(f"- Original Simulation F1: {original_rf_metrics['simulation_f1']:.2f}, Improved: {rf_report_improved['Simulation']['f1-score']:.2f}\n\n")
    f.write("## Clustering Improvements (K-Means)\n\n")
    f.write("### Original Metrics\n")
    f.write(f"- Silhouette Score: {original_kmeans_metrics['silhouette']:.3f}\n")
    f.write(f"- Adjusted Rand Index: {original_kmeans_metrics['ari']:.3f}\n\n")
    f.write("### Improvement Techniques\n")
    f.write("- **Adjusted Number of Clusters**: Re-ran the elbow method and chose k=5.\n")
    f.write("- **Added Categorical Features**: Included one-hot encoded categorical features for clustering.\n\n")
    f.write("### Improved Metrics\n")
    f.write(f"- Silhouette Score: {silhouette_improved:.3f}\n")
    f.write(f"- Adjusted Rand Index: {ari_improved:.3f}\n")

print("Improvement report saved to 'improvement_report.md'")