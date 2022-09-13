import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calc_f1(prec, recall):
    """Calculate the F1 Score

    Args:
        prec (float): The relevant precision
        recall (float): The relevant recall

    Returns:
        float: The F1 score.
    """
    return (2*prec*recall)/(prec + recall)

def calc_precision_recall(cm, cls):
    """Calculate precision and recall using a given confusion matrix.

    Args:
        cm (ndarray): The confusion matrix from which to calculate precision and recall
        cls (int): The index of the class to use as the positive class

    Returns:
        tuple: A tuple containing the precision and recall
    """
    # True positives
    tp = cm[cls, cls]
    
    # False negatives
    fn = np.sum(cm[cls, :]) - tp
    
    # False positives
    fp = np.sum(cm[:, cls]) - tp
    
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    
    return precision, recall

def plot_cm(cm):
    """Plot the given confusion matrix

    Args:
        cm (ndarray): The confusion matrix
    """
    # Inspired by https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#evaluate_metrics
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".2f")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
def process_labels(true, pred):
    """Process a set of true labels and predicted labels. Determine the predicted class from
    softmax outputs.

    Args:
        true (ndarray): True labels for samples
        pred (ndarray): Predicted labels for samples

    Returns:
        tuple: A tuple of the new true label array and the new predicted label array after processing.
    """
    new_t = []
    new_p = []
    
    for i in range(len(true)):
        new_t.append(true[i].argmax())
        new_p.append(pred[i].argmax())
        
    return np.array(new_t), np.array(new_p)