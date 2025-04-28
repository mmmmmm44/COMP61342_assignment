# evaluation sub-functions
# grab accuracy score, confusion matrix, and classification report

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def compute_accuracy(y_true, y_pred):
    """Compute accuracy score"""
    return accuracy_score(y_true, y_pred)

def compute_f1_score(y_true, y_pred):
    """Compute F1 score"""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def compute_balanced_accuracy(y_true, y_pred):
    """Compute balanced accuracy score"""
    return balanced_accuracy_score(y_true, y_pred, )


def compute_classification_report(y_true, y_pred, labels):
    """Compute classification report"""
    return classification_report(y_true, y_pred, target_names=labels, zero_division=0)

def compute_confusion_matrix(y_true, y_pred, labels, save=False, save_path=None):
    """Compute confusion matrix"""
    cm_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12,12), dpi=300)
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.xticks(rotation=90)
    fig = plt.gcf()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save is True")
        fig.savefig(save_path)

    plt.close(fig)