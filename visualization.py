import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import issparse

# Step 1: Load data and predictions
# Classification data
X_train = np.load('X_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

# Load predictions
knn_preds = np.load('knn_predictions.npy', allow_pickle=True)
nb_preds = np.load('nb_predictions.npy', allow_pickle=True)
rf_preds = np.load('rf_predictions.npy', allow_pickle=True)

# Clustering data
X_clustering_scaled = np.load('X_clustering_scaled.npy', allow_pickle=True)
kmeans_labels = np.load('kmeans_cluster_labels.npy', allow_pickle=True)

# Load improved clustering data (numerical + categorical features)
X_clustering_improved = np.load('X_clustering_improved.npy', allow_pickle=True)
kmeans_labels_improved = np.load('kmeans_labels_improved.npy', allow_pickle=True)

# Step 2: Handle 0D arrays and sparse matrices
# Unwrap 0D arrays for classification data
if isinstance(X_train, np.ndarray) and X_train.shape == ():
    X_train = X_train.item()
if isinstance(X_test, np.ndarray) and X_test.shape == ():
    X_test = X_test.item()

# Convert sparse matrices to dense for classification
if issparse(X_train):
    X_train = X_train.toarray()
if issparse(X_test):
    X_test = X_test.toarray()

# Unwrap 0D arrays for clustering data
if isinstance(X_clustering_scaled, np.ndarray) and X_clustering_scaled.shape == ():
    X_clustering_scaled = X_clustering_scaled.item()
if isinstance(X_clustering_improved, np.ndarray) and X_clustering_improved.shape == ():
    X_clustering_improved = X_clustering_improved.item()

# Step 3: Plot confusion matrices for classification models
labels = ['Action', 'Adventure', 'RPG', 'Simulation']

# KNN Confusion Matrix
cm_knn = confusion_matrix(y_test, knn_preds, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('static/images/knn_confusion_matrix.png')
plt.close()

# Naïve Bayes Confusion Matrix
cm_nb = confusion_matrix(y_test, nb_preds, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Naïve Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('static/images/nb_confusion_matrix.png')
plt.close()

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_preds, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('static/images/rf_confusion_matrix.png')
plt.close()

# Step 4: Generate classification reports and comparison bar charts
# Compute classification reports
knn_report = classification_report(y_test, knn_preds, labels=labels, output_dict=True)
nb_report = classification_report(y_test, nb_preds, labels=labels, output_dict=True)
rf_report = classification_report(y_test, rf_preds, labels=labels, output_dict=True)

# Organize metrics for plotting
metrics = {
    'KNN': knn_report,
    'NaiveBayes': nb_report,
    'RandomForest': rf_report
}

# Classes and algorithms
classes = ['Action', 'Adventure', 'RPG', 'Simulation']
algorithms = ['KNN', 'NaiveBayes', 'RandomForest']
metric_types = ['precision', 'recall', 'f1-score', 'support']

# Generate comparison bar charts for per-class metrics
for metric in metric_types:
    # Prepare data for grouped bar chart
    bar_width = 0.25  # Width of each bar
    x = np.arange(len(classes))  # Positions for the classes

    # Extract values for each algorithm
    knn_values = [metrics['KNN'][cls][metric] for cls in classes]
    nb_values = [metrics['NaiveBayes'][cls][metric] for cls in classes]
    rf_values = [metrics['RandomForest'][cls][metric] for cls in classes]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width, knn_values, bar_width, label='KNN', color='#ff9999')
    plt.bar(x, nb_values, bar_width, label='Naïve Bayes', color='#66b3ff')
    plt.bar(x + bar_width, rf_values, bar_width, label='Random Forest', color='#99ff99')

    # Add labels on top of bars
    for i in range(len(classes)):
        plt.text(x[i] - bar_width, knn_values[i], round(knn_values[i], 2) if metric != 'support' else int(knn_values[i]), 
                 ha='center', va='bottom')
        plt.text(x[i], nb_values[i], round(nb_values[i], 2) if metric != 'support' else int(nb_values[i]), 
                 ha='center', va='bottom')
        plt.text(x[i] + bar_width, rf_values[i], round(rf_values[i], 2) if metric != 'support' else int(rf_values[i]), 
                 ha='center', va='bottom')

    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Comparison Across Algorithms')
    plt.xticks(x, classes)
    plt.legend()
    plt.ylim(0, 1.0 if metric != 'support' else max(max(knn_values), max(nb_values), max(rf_values)) * 1.2)  # Adjust y-axis for support
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'static/images/comparison_{metric}_chart.png')
    plt.close()

# Generatemunition for accuracy comparison chart
accuracies = {
    'KNN': metrics['KNN']['accuracy'],
    'NaiveBayes': metrics['NaiveBayes']['accuracy'],
    'RandomForest': metrics['RandomForest']['accuracy']
}

plt.figure(figsize=(8, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#ff9999', '#66b3ff', '#99ff99'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Algorithms')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/images/comparison_accuracy_chart.png')
plt.close()

# Step 5: Plot scatter plots for K-Means clusters
# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)

# Original clustering (k=4, numerical features only)
X_pca = pca.fit_transform(X_clustering_scaled)
explained_variance = pca.explained_variance_ratio_.sum()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'Original K-Means Clusters (k=4, PCA, Explained Variance: {explained_variance:.2f})')
plt.savefig('static/images/kmeans_clusters_scatter.png')
plt.close()

# Improved clustering (k=5, numerical + categorical features)
X_pca_improved = pca.fit_transform(X_clustering_improved)
explained_variance_improved = pca.explained_variance_ratio_.sum()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_improved[:, 0], X_pca_improved[:, 1], c=kmeans_labels_improved, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'Improved K-Means Clusters (k=5, PCA, Explained Variance: {explained_variance_improved:.2f})')
plt.savefig('static/images/kmeans_clusters_scatter_improved.png')
plt.close()

# Step 6: Plot Random Forest feature importance
# Train Random Forest using X_train and y_train
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Define feature names (from Task 1 preprocessing)
numerical_cols = ['original_price', 'discount_price', 'achievements', 'release_year', 
                  'num_languages', 'num_tags', 'sentiment_score', 'desc_word_count', 'requires_gpu']
categorical_cols = ['developer', 'publisher', 'popular_tags', 'game_details', 'mature_content']
cat_features = []
for col in categorical_cols:
    cat_features.extend([f"{col}_{i}" for i in range(10)])  # Simplified naming
feature_names = numerical_cols + cat_features

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15 features for clarity

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance (Top 15)')
plt.tight_layout()
plt.savefig('static/images/rf_feature_importance.png')
plt.close()

# Step 7: Save performance metrics to performance_report.md with proper formatting
with open('performance_report.md', 'w') as f:
    f.write("# Classification Metrics\n\n")

    # KNN
    f.write("## KNN\n")
    f.write("**Per-Class Metrics**\n")
    f.write("| Class      | Precision | Recall | F1-Score | Support |\n")
    f.write("|------------|-----------|--------|----------|---------|\n")
    for cls in classes:
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
    for cls in classes:
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
    for cls in classes:
        f.write(f"| {cls:<10} | {rf_report[cls]['precision']:.2f}      | {rf_report[cls]['recall']:.2f}   | {rf_report[cls]['f1-score']:.2f}    | {int(rf_report[cls]['support']):<7} |\n")
    f.write("\n")
    f.write(f"**Accuracy**: {rf_report['accuracy']:.2f}\n")
    f.write(f"**Macro Avg Precision**: {rf_report['macro avg']['precision']:.2f}, Recall: {rf_report['macro avg']['recall']:.2f}, F1-Score: {rf_report['macro avg']['f1-score']:.2f}\n")
    f.write(f"**Weighted Avg Precision**: {rf_report['weighted avg']['precision']:.2f}, Recall: {rf_report['weighted avg']['recall']:.2f}, F1-Score: {rf_report['weighted avg']['f1-score']:.2f}\n")

print("Performance metrics saved to 'performance_report.md'")
print("Visualizations saved as PNG files:")
print("- knn_confusion_matrix.png")
print("- nb_confusion_matrix.png")
print("- rf_confusion_matrix.png")
print("- kmeans_clusters_scatter.png")
print("- kmeans_clusters_scatter_improved.png")
print("- rf_feature_importance.png")
print("- comparison_accuracy_chart.png")
print("Comparison bar charts for metrics saved as:")
for metric in metric_types:
    print(f"- comparison_{metric}_chart.png")