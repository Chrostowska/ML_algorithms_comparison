# Advanced Machine Learning Pipeline

A comprehensive machine learning pipeline with advanced preprocessing, model selection, and evaluation capabilities.

## Features

### 1. Advanced Preprocessing (`preprocessing.py`)
- Automatic handling of numerical and categorical features
- Multiple scaling options (Standard, MinMax, Robust)
- Categorical encoding (Label, One-Hot)
- Missing value imputation (Mean, Median, KNN)
- Outlier detection and handling
- Feature selection
- Dimensionality reduction (PCA)

### 2. Model Factory (`models.py`)
- Unified interface for multiple ML algorithms
- Pre-configured hyperparameter search spaces
- Support for:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines
  - K-Nearest Neighbors
  - Gradient Boosting
  - Neural Networks
  - Naive Bayes
  - XGBoost

### 3. Evaluation and Visualization
- Comprehensive model evaluation metrics
- Performance visualization
- Feature importance analysis
- Learning curves
- ROC curves and AUC scores

## Project Structure
```
ml_project/
│
├── data/                    # Data storage directory
│   └── README.md           # Data directory documentation
│
├── ml_pipeline/            # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── preprocessing.py    # Data preprocessing utilities
│   ├── models.py          # Model definitions and configurations
│   ├── evaluation.py      # Evaluation metrics and visualization
│   └── utils.py           # Utility functions
│
├── notebooks/              # Jupyter notebooks for exploration
│   └── examples.ipynb     # Example usage notebook
│
├── tests/                  # Test directory
│   └── __init__.py        # Test initialization
│
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from ml_pipeline.preprocessing import DataPreprocessor
from ml_pipeline.models import ModelFactory

# Initialize preprocessor
preprocessor = DataPreprocessor(
    scaling_method='standard',
    handle_outliers=True,
    feature_selection=10
)

# Preprocess data
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)

# Create and train model
model = ModelFactory.create_model('random_forest')
model.fit(X_train_processed, y_train)

# Make predictions
predictions = model.predict(X_test_processed)
```

### Example Notebook

Check out `notebooks/examples.ipynb` for a complete example of:
- Data preprocessing
- Model training and evaluation
- Performance visualization
- Feature importance analysis

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 