"""
Credit Card Anomaly Detection Module

This module contains reusable functions for credit card fraud detection
using various anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score


def load_credit_card_data(filepath):
    """
    Load credit card transaction data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing credit card data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing credit card transaction data
    """
    return pd.read_csv(filepath, sep=',')


def get_fraud_statistics(data):
    """
    Calculate fraud statistics from the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing credit card data with 'Class' column
        
    Returns:
    --------
    dict
        Dictionary containing fraud and normal transaction counts
    """
    fraud = data[data['Class'] == 1]
    normal = data[data['Class'] == 0]
    
    return {
        'fraud_count': len(fraud),
        'normal_count': len(normal),
        'fraud_fraction': len(fraud) / float(len(normal)) if len(normal) > 0 else 0
    }


def prepare_features(data, target_column='Class'):
    """
    Prepare features and target for anomaly detection.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing credit card data
    target_column : str, optional
        Name of the target column (default: 'Class')
        
    Returns:
    --------
    tuple
        (X, Y) where X is feature matrix and Y is target vector
    """
    columns = data.columns.tolist()
    columns = [c for c in columns if c not in [target_column]]
    
    X = data[columns]
    Y = data[target_column]
    
    return X, Y


def create_anomaly_detectors(outlier_fraction, random_state=42):
    """
    Create anomaly detection classifiers.
    
    Parameters:
    -----------
    outlier_fraction : float
        Expected fraction of outliers in the dataset
    random_state : int, optional
        Random state for reproducibility (default: 42)
        
    Returns:
    --------
    dict
        Dictionary of classifier names and instances
    """
    state = np.random.RandomState(random_state)
    
    classifiers = {
        "Isolation Forest": IsolationForest(
            n_estimators=100,
            contamination=outlier_fraction,
            random_state=state,
            verbose=0
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20,
            algorithm='auto',
            leaf_size=30,
            metric='minkowski',
            p=2,
            metric_params=None,
            contamination=outlier_fraction
        ),
        "Support Vector Machine": OneClassSVM(
            kernel='rbf',
            degree=3,
            gamma=0.1,
            nu=0.05,
            max_iter=-1
        )
    }
    
    return classifiers


def train_and_predict(clf, clf_name, X):
    """
    Train classifier and make predictions.
    
    Parameters:
    -----------
    clf : object
        Classifier instance
    clf_name : str
        Name of the classifier
    X : pd.DataFrame or np.ndarray
        Feature matrix
        
    Returns:
    --------
    np.ndarray
        Predictions (1 for fraud, 0 for normal)
    """
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
    else:
        clf.fit(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for Valid transactions, 1 for Fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    return y_pred


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    n_errors = (y_pred != y_true).sum()
    acc_score = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'n_errors': n_errors,
        'accuracy_score': acc_score,
        'classification_report': class_report
    }
