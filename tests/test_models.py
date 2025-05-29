"""
Tests for the models module and model factory.
"""

import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from ml_pipeline.model_factory import ModelFactory

def test_model_factory_initialization():
    """Test that ModelFactory can be initialized."""
    factory = ModelFactory()
    assert factory is not None

@pytest.mark.parametrize("model_type", [
    'logistic',
    'rf',
    'svm',
    'knn',
    'dt'
])
def test_model_creation(model_type, model_parameters):
    """Test creation of different model types."""
    factory = ModelFactory()
    model = factory.create_model(model_type, **model_parameters[model_type])
    assert model is not None

def test_invalid_model_type():
    """Test that invalid model type raises appropriate error."""
    factory = ModelFactory()
    with pytest.raises(ValueError):
        factory.create_model('invalid_model_type')

def test_model_training(sample_data):
    """Test that models can be trained successfully."""
    X, y = sample_data
    factory = ModelFactory()
    
    for model_type in ['logistic', 'rf', 'svm', 'knn', 'dt']:
        model = factory.create_model(model_type)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(isinstance(pred, (np.int64, np.int32, int)) for pred in predictions)

def test_model_cross_validation(breast_cancer_data):
    """Test cross-validation functionality."""
    X, y = breast_cancer_data
    factory = ModelFactory()
    
    for model_type in ['logistic', 'rf', 'svm', 'knn', 'dt']:
        model = factory.create_model(model_type)
        
        # Ensure the model can be used in cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5)
        
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)

def test_model_prediction_without_training(sample_data):
    """Test that prediction without training raises appropriate error."""
    X, _ = sample_data
    factory = ModelFactory()
    
    for model_type in ['logistic', 'rf', 'svm', 'knn', 'dt']:
        model = factory.create_model(model_type)
        with pytest.raises(NotFittedError):
            model.predict(X)

def test_model_parameter_validation():
    """Test that invalid parameters are handled appropriately."""
    factory = ModelFactory()
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        factory.create_model('logistic', C=-1.0)  # C must be positive
    
    with pytest.raises(ValueError):
        factory.create_model('knn', n_neighbors=0)  # n_neighbors must be positive

@pytest.mark.parametrize("model_type,expected_params", [
    ('logistic', {'C': 1.0, 'max_iter': 1000}),
    ('rf', {'n_estimators': 100, 'max_depth': None}),
    ('svm', {'C': 1.0, 'kernel': 'rbf'}),
    ('knn', {'n_neighbors': 5}),
    ('dt', {'max_depth': None})
])
def test_default_parameters(model_type, expected_params):
    """Test that default parameters are set correctly."""
    factory = ModelFactory()
    params = factory.get_default_params(model_type)
    assert params == expected_params 