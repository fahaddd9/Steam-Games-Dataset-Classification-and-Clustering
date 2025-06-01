import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, silhouette_score, adjusted_rand_score

# Step 1: Load data and predictions
# Classification data
y_test = np.load('y_test.npy', allow_pickle=True)
knn_preds = np.load('knn_predictions.npy', allow_pickle=True)
nb_preds = np.load('nb_predictions.npy', allow_pickle=True)
rf_preds = np.load('rf_predictions.npy', allow_pickle=True)

# Clustering data
X_clustering_scaled = np.load('X_clustering_scaled.npy', allow_pickle=True)
kmeans_labels = np.load('kmeans_cluster_labels.npy', allow_pickle=True)

# Load true labels for clustering evaluation
df = pd.read_csv('preprocessed_steam_games.csv')
y_true_all = df['primary_genre'].values

# Step 2: Evaluate classification models
labels = ['Action', 'Adventure', 'RPG', 'Simulation']

# Compute classification reports
knn_report = classification_report(y_test, knn_preds, labels=labels, output_dict=True)
nb_report = classification_report(y_test, nb_preds, labels=labels, output_dict=True)
rf_report = classification_report(y_test, rf_preds, labels=labels, output_dict=True)

# Step 3: Evaluate clustering
silhouette = silhouette_score(X_clustering_scaled, kmeans_labels)
ari = adjusted_rand_score(y_true_all, kmeans_labels)

# Step 4: Save performance metrics to performance_report.md with proper formatting
with open('performance_report.md', 'w') as f:
    f.write("# Classification Metrics\n\n")

    # KNN
    f.write("## KNN\n")
    f.write("**Per-Class Metrics**\n")
    f.write("| Class      | Precision | Recall | F1-Score | Support |\n")
    f.write("|------------|-----------|--------|----------|---------|\n")
    for cls in labels:
        f.write(f"| {cls:<10} | {knn_report[cls]['precision']:.2f}      | {knn_report[cls]['recall']:.2f}   | {knn_report[cls]['f1-score']:.2f}    | {int(knn_report[cls]['support']):<7} |\n")
    f.write("\n")
    f.write(f"**Accuracy**: {knn_report['accuracy']:.2f}\n")
    f.write(f"**Macro Avg Precision**: {knn_report['macro avg']['precision']:.2f}, Recall: {knn_report['macro avg']['recall']:.2f}, F1-Score: {knn_report['macro avg']['f1-score']:.2f}\n")
    f.write(f"**Weighted Avg Precision**: {knn_report['weighted avg']['precision']:.2f}, Recall: {knn_report['weighted avg']['recall']:.2f}, F1-Score: {knn_report['weighted avg']['f1-score']:.2f}\n\n")

    # Naïve Bayes
    f.write("## Naïve Bayes\n")
    f.write("**Per-Class Metrics**\n")
    f.write("| Class      | Precision | Recall | F1-Score | Support |\n")
    f.write("|------------|-----------|--------|----------|---------|\n")
    for cls in labels:
        f.write(f"| {cls:<10} | {nb_report[cls]['precision']:.2f}      | {nb_report[cls]['recall']:.2f}   | {nb_report[cls]['f1-score']:.2f}    | {int(nb_report[cls]['support']):<7} |\n")
    f.write("\n")
    f.write(f"**Accuracy**: {nb_report['accuracy']:.2f}\n")
    f.write(f"**Macro Avg Precision**: {nb_report['macro avg']['precision']:.2f}, Recall: {nb_report['macro avg']['recall']:.2f}, F1-Score: {nb_report['macro avg']['f1-score']:.2f}\n")
    f.write(f"**Weighted Avg Precision**: {nb_report['weighted avg']['precision']:.2f}, Recall: {nb_report['weighted avg']['recall']:.2f}, F1-Score: {nb_report['weighted avg']['f1-score']:.2f}\n\n")

    # Random Forest
    f.write("## Random Forest\n")
    f.write("**Per-Class Metrics**\n")
    f.write("| Class      | Precision | Recall | F1-Score | Support |\n")
    f.write("|------------|-----------|--------|----------|---------|\n")
    for cls in labels:
        f.write(f"| {cls:<10} | {rf_report[cls]['precision']:.2f}      | {rf_report[cls]['recall']:.2f}   | {rf_report[cls]['f1-score']:.2f}    | {int(rf_report[cls]['support']):<7} |\n")
    f.write("\n")
    f.write(f"**Accuracy**: {rf_report['accuracy']:.2f}\n")
    f.write(f"**Macro Avg Precision**: {rf_report['macro avg']['precision']:.2f}, Recall: {rf_report['macro avg']['recall']:.2f}, F1-Score: {rf_report['macro avg']['f1-score']:.2f}\n")
    f.write(f"**Weighted Avg Precision**: {rf_report['weighted avg']['precision']:.2f}, Recall: {rf_report['weighted avg']['recall']:.2f}, F1-Score: {rf_report['weighted avg']['f1-score']:.2f}\n\n")

    # Clustering Metrics
    f.write("## Clustering Metrics (K-Means, k=4)\n")
    f.write(f"- Silhouette Score: {silhouette:.3f}\n")
    f.write(f"- Adjusted Rand Index: {ari:.3f}\n")

print("Performance metrics saved to 'performance_report.md'")