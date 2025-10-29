"""
Unit tests for anomaly detection module
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anomaly_detection import (
    load_credit_card_data,
    get_fraud_statistics,
    prepare_features,
    create_anomaly_detectors,
    train_and_predict,
    evaluate_model
)


@pytest.fixture
def sample_data():
    """Create sample credit card data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'Time': np.random.rand(n_samples) * 100,
        'V1': np.random.randn(n_samples),
        'V2': np.random.randn(n_samples),
        'V3': np.random.randn(n_samples),
        'Amount': np.random.rand(n_samples) * 1000,
        'Class': np.concatenate([np.zeros(90), np.ones(10)])  # 10% fraud
    })
    
    return data


@pytest.fixture
def sample_csv_file(tmp_path, sample_data):
    """Create a temporary CSV file with sample data"""
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    return str(csv_file)


def test_load_credit_card_data(sample_csv_file):
    """Test loading credit card data from CSV"""
    data = load_credit_card_data(sample_csv_file)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'Class' in data.columns


def test_get_fraud_statistics(sample_data):
    """Test fraud statistics calculation"""
    stats = get_fraud_statistics(sample_data)
    
    assert 'fraud_count' in stats
    assert 'normal_count' in stats
    assert 'fraud_fraction' in stats
    assert stats['fraud_count'] == 10
    assert stats['normal_count'] == 90
    assert stats['fraud_fraction'] > 0


def test_get_fraud_statistics_empty_normal():
    """Test fraud statistics with no normal transactions"""
    data = pd.DataFrame({'Class': [1, 1, 1]})
    stats = get_fraud_statistics(data)
    
    assert stats['fraud_count'] == 3
    assert stats['normal_count'] == 0
    assert stats['fraud_fraction'] == 0


def test_prepare_features(sample_data):
    """Test feature preparation"""
    X, Y = prepare_features(sample_data)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.Series)
    assert 'Class' not in X.columns
    assert len(X) == len(Y)
    assert len(X) == 100


def test_prepare_features_custom_target(sample_data):
    """Test feature preparation with custom target column"""
    sample_data['Target'] = sample_data['Class']
    X, Y = prepare_features(sample_data, target_column='Target')
    
    assert 'Target' not in X.columns
    assert 'Class' in X.columns


def test_create_anomaly_detectors():
    """Test creation of anomaly detectors"""
    outlier_fraction = 0.1
    classifiers = create_anomaly_detectors(outlier_fraction, random_state=42)
    
    assert isinstance(classifiers, dict)
    assert len(classifiers) == 3
    assert "Isolation Forest" in classifiers
    assert "Local Outlier Factor" in classifiers
    assert "Support Vector Machine" in classifiers
    
    assert isinstance(classifiers["Isolation Forest"], IsolationForest)
    assert isinstance(classifiers["Local Outlier Factor"], LocalOutlierFactor)
    assert isinstance(classifiers["Support Vector Machine"], OneClassSVM)


def test_create_anomaly_detectors_parameters():
    """Test that anomaly detectors are created with correct parameters"""
    outlier_fraction = 0.15
    classifiers = create_anomaly_detectors(outlier_fraction, random_state=123)
    
    iso_forest = classifiers["Isolation Forest"]
    assert iso_forest.contamination == outlier_fraction
    assert iso_forest.n_estimators == 100


def test_train_and_predict_isolation_forest(sample_data):
    """Test training and prediction with Isolation Forest"""
    X, Y = prepare_features(sample_data)
    clf = IsolationForest(contamination=0.1, random_state=42)
    
    y_pred = train_and_predict(clf, "Isolation Forest", X)
    
    assert len(y_pred) == len(X)
    assert set(y_pred).issubset({0, 1})


def test_train_and_predict_lof(sample_data):
    """Test training and prediction with Local Outlier Factor"""
    X, Y = prepare_features(sample_data)
    clf = LocalOutlierFactor(contamination=0.1)
    
    y_pred = train_and_predict(clf, "Local Outlier Factor", X)
    
    assert len(y_pred) == len(X)
    assert set(y_pred).issubset({0, 1})


def test_train_and_predict_svm(sample_data):
    """Test training and prediction with One-Class SVM"""
    X, Y = prepare_features(sample_data)
    clf = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
    
    y_pred = train_and_predict(clf, "Support Vector Machine", X)
    
    assert len(y_pred) == len(X)
    assert set(y_pred).issubset({0, 1})


def test_evaluate_model():
    """Test model evaluation"""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    
    metrics = evaluate_model(y_true, y_pred)
    
    assert 'n_errors' in metrics
    assert 'accuracy_score' in metrics
    assert 'classification_report' in metrics
    assert metrics['n_errors'] == 2
    assert 0 <= metrics['accuracy_score'] <= 1
    assert isinstance(metrics['classification_report'], dict)


def test_evaluate_model_perfect_prediction():
    """Test evaluation with perfect predictions"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    metrics = evaluate_model(y_true, y_pred)
    
    assert metrics['n_errors'] == 0
    assert metrics['accuracy_score'] == 1.0


def test_integration_full_pipeline(sample_data):
    """Test complete pipeline from data to evaluation"""
    # Prepare features
    X, Y = prepare_features(sample_data)
    
    # Get fraud statistics
    stats = get_fraud_statistics(sample_data)
    outlier_fraction = stats['fraud_fraction']
    
    # Create classifiers
    classifiers = create_anomaly_detectors(outlier_fraction, random_state=42)
    
    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        y_pred = train_and_predict(clf, clf_name, X)
        metrics = evaluate_model(Y, y_pred)
        
        # Check that we got valid results
        assert metrics['n_errors'] >= 0
        assert 0 <= metrics['accuracy_score'] <= 1
        assert 'classification_report' in metrics
