import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

file_path = './results/prelim_results.tsv'

data = pd.read_csv(file_path, sep='\t')

data['predicted'] = data['predicted'].apply(lambda x: x.split('_')[1]).astype(int)

def calculate_metrics(target, predicted):
    """
    Calculate Precision, Recall, F1 Score, and accuracy 
    
    Parameters:
    - target: true labels
    - predicted: predicted labels
    
    Returns:
    A dictionary containing precision, recall, F1 score. and accuracy
    """
    precision = precision_score(target, predicted, average='macro')
    recall = recall_score(target, predicted, average='macro')
    f1 = f1_score(target, predicted, average='macro')
    accuracy = accuracy_score(target, predicted)
    
    return {
        'Precision': precision, #correct prediction out of all positive predictions
        'Recall': recall, #correct prediction out of all actual positive instances
        'F1-Score': f1, #"harmonic mean" of precision and recall. Gives single measure of performance using precision and recall
        'Accuracy': accuracy #correct predictions (true positives and true negatives) out of all predictions 
    }

metrics = calculate_metrics(data['target'], data['predicted'])
print(metrics)
