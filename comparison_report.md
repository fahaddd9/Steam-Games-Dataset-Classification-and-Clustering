# Model Comparison

## KNN
**Per-Class Metrics**
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Action     | 0.74      | 0.88   | 0.80    | 3277    |
| Adventure  | 0.56      | 0.39   | 0.46    | 1407    |
| RPG        | 0.68      | 0.30   | 0.42    | 178     |
| Simulation | 0.90      | 0.65   | 0.75    | 367     |

**Accuracy**: 0.71
**Macro Avg Precision**: 0.72, Recall: 0.56, F1-Score: 0.61
**Weighted Avg Precision**: 0.70, Recall: 0.71, F1-Score: 0.70

## Naïve Bayes
**Per-Class Metrics**
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Action     | 1.00      | 0.19   | 0.31    | 3277    |
| Adventure  | 0.54      | 0.32   | 0.40    | 1407    |
| RPG        | 0.08      | 0.35   | 0.13    | 178     |
| Simulation | 0.11      | 0.90   | 0.19    | 367     |

**Accuracy**: 0.28
**Macro Avg Precision**: 0.43, Recall: 0.44, F1-Score: 0.26
**Weighted Avg Precision**: 0.78, Recall: 0.28, F1-Score: 0.32

## Random Forest
**Per-Class Metrics**
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Action     | 0.76      | 0.91   | 0.83    | 3277    |
| Adventure  | 0.66      | 0.44   | 0.53    | 1407    |
| RPG        | 0.67      | 0.39   | 0.49    | 178     |
| Simulation | 0.89      | 0.71   | 0.79    | 367     |

**Accuracy**: 0.75
**Macro Avg Precision**: 0.75, Recall: 0.61, F1-Score: 0.66
**Weighted Avg Precision**: 0.74, Recall: 0.75, F1-Score: 0.73

## Summary
- **Random Forest** outperformed the other models with an accuracy of 0.75 and a macro F1-score of 0.66, particularly excelling on minority classes (RPG F1: 0.49, Simulation F1: 0.79).
- **KNN** achieved an accuracy of 0.71 and a macro F1-score of 0.61, performing well on majority classes but struggling with minority classes (RPG F1: 0.42, Simulation F1: 0.75).
- **Naïve Bayes** had the lowest performance with an accuracy of 0.28 and a macro F1-score of 0.26, struggling significantly with minority classes (RPG F1: 0.13, Simulation F1: 0.19).
