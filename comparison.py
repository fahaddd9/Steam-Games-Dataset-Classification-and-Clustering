import numpy as np
from sklearn.metrics import classification_report

# Step 1: Load predictions and test labels
y_test = np.load('y_test.npy', allow_pickle=True)
knn_preds = np.load('knn_predictions.npy', allow_pickle=True)
nb_preds = np.load('nb_predictions.npy', allow_pickle=True)
rf_preds = np.load('rf_predictions.npy', allow_pickle=True)

# Step 2: Compute classification reports
labels = ['Action', 'Adventure', 'RPG', 'Simulation']
knn_report = classification_report(y_test, knn_preds, labels=labels, output_dict=True)
nb_report = classification_report(y_test, nb_preds, labels=labels, output_dict=True)
rf_report = classification_report(y_test, rf_preds, labels=labels, output_dict=True)

# Step 3: Write comparison report
with open('comparison_report.md', 'w') as f:
    f.write("# Model Comparison\n\n")

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

    f.write("## Summary\n")
    f.write(f"- **Random Forest** outperformed the other models with an accuracy of {rf_report['accuracy']:.2f} and a macro F1-score of {rf_report['macro avg']['f1-score']:.2f}, particularly excelling on minority classes (RPG F1: {rf_report['RPG']['f1-score']:.2f}, Simulation F1: {rf_report['Simulation']['f1-score']:.2f}).\n")
    f.write(f"- **KNN** achieved an accuracy of {knn_report['accuracy']:.2f} and a macro F1-score of {knn_report['macro avg']['f1-score']:.2f}, performing well on majority classes but struggling with minority classes (RPG F1: {knn_report['RPG']['f1-score']:.2f}, Simulation F1: {knn_report['Simulation']['f1-score']:.2f}).\n")
    f.write(f"- **Naïve Bayes** had the lowest performance with an accuracy of {nb_report['accuracy']:.2f} and a macro F1-score of {nb_report['macro avg']['f1-score']:.2f}, struggling significantly with minority classes (RPG F1: {nb_report['RPG']['f1-score']:.2f}, Simulation F1: {nb_report['Simulation']['f1-score']:.2f}).\n")

print("Comparison report saved to 'comparison_report.md'")