"""
Tests for the preprocessing module.
"""

import pytest
import numpy as np
from ml_pipeline.preprocessing import DataPreprocessor

def test_preprocessor_initialization():
    """Test that DataPreprocessor initializes with default parameters."""
    preprocessor = DataPreprocessor()
    assert preprocessor is not None
    assert hasattr(preprocessor, 'scaler')
    assert hasattr(preprocessor, 'feature_selector')

def test_data_scaling(sample_data):
    """Test that data scaling works correctly."""
    X, _ = sample_data
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor._scale_features(X)
    
    # Check that scaling resulted in zero mean and unit variance
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)

def test_train_test_split(sample_data):
    """Test that train-test splitting works correctly."""
    X, y = sample_data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, test_size=0.25)
    
    # Check shapes
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1

def test_handle_missing_values(sample_dataframe):
    """Test handling of missing values."""
    # Create data with missing values
    df = sample_dataframe.copy()
    df.iloc[0, 0] = np.nan
    
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor._handle_missing_values(df)
    
    assert not df_cleaned.isnull().any().any()

def test_feature_selection(breast_cancer_data):
    """Test feature selection functionality."""
    X, y = breast_cancer_data
    preprocessor = DataPreprocessor()
    X_selected = preprocessor._select_features(X, y, n_features=10)
    
    # Check that we get the requested number of features
    assert X_selected.shape[1] == 10

def test_full_preprocessing_pipeline(breast_cancer_data):
    """Test the complete preprocessing pipeline."""
    X, y = breast_cancer_data
    preprocessor = DataPreprocessor()
    
    # Process the data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y)
    
    # Verify the output
    assert X_train.shape[0] + X_test.shape[0] == len(X)
    assert X_train.shape[1] == X_test.shape[1]
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

@pytest.mark.parametrize("invalid_input", [
    None,
    [[1, 2], [3, 4]],  # List instead of numpy array
    np.array([]),  # Empty array
])
def test_invalid_input_handling(invalid_input):
    """Test that invalid inputs are handled appropriately."""
    preprocessor = DataPreprocessor()
    with pytest.raises((ValueError, TypeError)):
        preprocessor.prepare_data(invalid_input, np.array([1, 0])) 