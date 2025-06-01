# Task 7: Efforts to Improve Results

## Classification Improvements

### KNN
- Original Accuracy: 0.65, Improved: 0.62
- Original Macro F1: 0.48, Improved: 0.53
- Original RPG F1: 0.32, Improved: 0.30
- Original Simulation F1: 0.45, Improved: 0.61

### Naïve Bayes
- Original Accuracy: 0.68, Improved: 0.58
- Original Macro F1: 0.50, Improved: 0.51
- Original RPG F1: 0.35, Improved: 0.16
- Original Simulation F1: 0.47, Improved: 0.71

### Random Forest
- Original Accuracy: 0.69, Improved: 0.72
- Original Macro F1: 0.52, Improved: 0.63
- Original RPG F1: 0.36, Improved: 0.45
- Original Simulation F1: 0.48, Improved: 0.76

## Clustering Improvements (K-Means)

### Original Metrics
- Silhouette Score: 0.342
- Adjusted Rand Index: 0.125

### Improvement Techniques
- **Adjusted Number of Clusters**: Re-ran the elbow method and chose k=5.
- **Added Categorical Features**: Included one-hot encoded categorical features for clustering.

### Improved Metrics
- Silhouette Score: 0.184
- Adjusted Rand Index: 0.000
