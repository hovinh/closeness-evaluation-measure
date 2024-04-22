import numpy as np

def compute_accuracy_score(confusion_matrix: np.array) -> float:
    true_positives = np.diag(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    accuracy = np.sum(true_positives) / total_samples

    return accuracy