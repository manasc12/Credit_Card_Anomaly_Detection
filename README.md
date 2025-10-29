# Credit Card Anomaly Detection

[![Testing Pipeline](https://github.com/manasc12/Credit_Card_Anomaly_Detection/actions/workflows/test.yml/badge.svg)](https://github.com/manasc12/Credit_Card_Anomaly_Detection/actions/workflows/test.yml)

A comprehensive credit card fraud detection system using machine learning anomaly detection algorithms. This project implements and compares three different anomaly detection techniques: Isolation Forest, Local Outlier Factor, and One-Class SVM.

## Features

- 🔍 **Multiple Anomaly Detection Algorithms**: Compare performance of different ML approaches
- 📊 **Data Visualization**: Comprehensive EDA and results visualization
- 🧪 **Complete Testing Pipeline**: Automated testing with 100% code coverage
- 🔄 **CI/CD Integration**: GitHub Actions for continuous testing
- 📦 **Modular Code**: Reusable functions for reproducibility

## Algorithms Implemented

1. **Isolation Forest**: Ensemble-based anomaly detection
2. **Local Outlier Factor (LOF)**: Density-based outlier detection
3. **One-Class SVM**: Support Vector Machine for novelty detection

## Project Structure

```
Credit_Card_Anomaly_Detection/
├── Anamoly Detection.ipynb      # Main analysis notebook
├── anomaly_detection.py         # Reusable detection functions
├── creditcard_data_mini.csv     # Sample dataset
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── .coveragerc                  # Coverage configuration
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_anomaly_detection.py
└── .github/
    └── workflows/
        └── test.yml             # CI/CD pipeline
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/manasc12/Credit_Card_Anomaly_Detection.git
cd Credit_Card_Anomaly_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

Launch Jupyter and open the main notebook:
```bash
jupyter notebook "Anamoly Detection.ipynb"
```

### Using the Module

You can also use the anomaly detection functions programmatically:

```python
from anomaly_detection import (
    load_credit_card_data,
    get_fraud_statistics,
    prepare_features,
    create_anomaly_detectors,
    train_and_predict,
    evaluate_model
)

# Load data
data = load_credit_card_data('creditcard_data_mini.csv')

# Prepare features
X, Y = prepare_features(data)

# Get fraud statistics
stats = get_fraud_statistics(data)
outlier_fraction = stats['fraud_fraction']

# Create and train models
classifiers = create_anomaly_detectors(outlier_fraction)
for clf_name, clf in classifiers.items():
    y_pred = train_and_predict(clf, clf_name, X)
    metrics = evaluate_model(Y, y_pred)
    print(f"{clf_name}: Accuracy = {metrics['accuracy_score']:.2f}")
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Tests with Coverage

```bash
pytest tests/ -v --cov=. --cov-report=html
```

### Test Notebook Execution

```bash
pytest --nbmake "Anamoly Detection.ipynb" --nbmake-timeout=300
```

### Run Complete Test Suite

```bash
# Unit tests + notebook tests
pytest tests/ --nbmake "Anamoly Detection.ipynb" -v --cov=. --cov-report=term-missing
```

## Continuous Integration

This project uses GitHub Actions for automated testing. Every push and pull request to the main branch triggers:

- ✅ Unit tests across Python 3.8, 3.9, and 3.10
- ✅ Notebook execution validation
- ✅ Code coverage reporting
- ✅ Dependency validation

View the workflow status in the [Actions tab](https://github.com/manasc12/Credit_Card_Anomaly_Detection/actions).

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders.

- **Features**: V1-V28 (PCA transformed), Time, Amount
- **Target**: Class (0 = Normal, 1 = Fraud)
- **Balance**: Highly imbalanced dataset

[Original Dataset Link](https://www.kaggle.com/naveengowda16/anomaly-detection-credit-card-fraud-analysis/data)

## Model Performance

The notebook evaluates each algorithm using:
- **Accuracy Score**: Overall correctness
- **Classification Report**: Precision, recall, F1-score
- **Confusion Matrix**: True positives/negatives analysis

Due to the imbalanced nature of the dataset, we primarily focus on precision-recall metrics rather than overall accuracy.

## Development

### Adding New Tests

1. Create test functions in `tests/test_anomaly_detection.py`
2. Follow the naming convention: `test_<function_name>`
3. Use pytest fixtures for common test data
4. Run tests locally before committing

### Code Coverage Goals

- Target: 100% coverage for core functions
- Current: 100% coverage achieved ✅
- View detailed report: `pytest --cov=. --cov-report=html` then open `htmlcov/index.html`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Dependencies

Core dependencies:
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.1` - Machine learning algorithms
- `scipy>=1.8.0` - Scientific computing
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization

Testing dependencies:
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=3.0.0` - Coverage reporting
- `nbmake>=1.3.0` - Notebook testing
- `nbval>=0.9.6` - Notebook validation

See `requirements.txt` for complete list.

## License

This project is available for educational and research purposes.

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/)
- Inspired by real-world credit card fraud detection challenges
- Built with industry-standard ML and testing practices

## Contact

For questions or feedback, please open an issue on GitHub.