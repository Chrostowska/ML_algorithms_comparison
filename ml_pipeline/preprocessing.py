import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

class DataPreprocessor:
    """
    A comprehensive data preprocessing class that handles various preprocessing tasks.
    """
    def __init__(self, 
                 scaling_method='standard',
                 categorical_encoding='label',
                 imputation_method='mean',
                 feature_selection=None,
                 n_components=None,
                 handle_outliers=False):
        """
        Initialize the preprocessor with specified methods.
        
        Parameters:
        -----------
        scaling_method : str, optional (default='standard')
            Method for scaling numerical features.
            Options: 'standard', 'minmax', 'robust', None
        
        categorical_encoding : str, optional (default='label')
            Method for encoding categorical variables.
            Options: 'label', 'onehot', None
            
        imputation_method : str, optional (default='mean')
            Method for handling missing values.
            Options: 'mean', 'median', 'most_frequent', 'knn', None
            
        feature_selection : int or float, optional (default=None)
            Number of features to select (if int) or proportion of features (if float)
            
        n_components : int or float, optional (default=None)
            Number of components for PCA (if int) or variance to preserve (if float)
            
        handle_outliers : bool, optional (default=False)
            Whether to handle outliers using IQR method
        """
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.imputation_method = imputation_method
        self.feature_selection = feature_selection
        self.n_components = n_components
        self.handle_outliers = handle_outliers
        
        # Initialize components
        self.num_scaler = None
        self.cat_encoder = None
        self.imputer = None
        self.feature_selector = None
        self.pca = None
        self.numerical_features = None
        self.categorical_features = None
        
    def _identify_feature_types(self, X):
        """Identify numerical and categorical columns."""
        if isinstance(X, pd.DataFrame):
            self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns
        else:
            self.numerical_features = np.arange(X.shape[1])
            self.categorical_features = []
            
    def _initialize_components(self):
        """Initialize preprocessing components based on specified methods."""
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.num_scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.num_scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.num_scaler = RobustScaler()
            
        # Initialize encoder
        if self.categorical_encoding == 'label':
            self.cat_encoder = LabelEncoder()
        elif self.categorical_encoding == 'onehot':
            self.cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
        # Initialize imputer
        if self.imputation_method == 'knn':
            self.imputer = KNNImputer()
        elif self.imputation_method:
            self.imputer = SimpleImputer(strategy=self.imputation_method)
            
        # Initialize feature selector
        if self.feature_selection:
            self.feature_selector = SelectKBest(score_func=f_classif, 
                                              k=self.feature_selection)
            
        # Initialize PCA
        if self.n_components:
            self.pca = PCA(n_components=self.n_components)
            
    def _handle_outliers(self, X):
        """Handle outliers using IQR method."""
        if isinstance(X, pd.DataFrame):
            for column in self.numerical_features:
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[column] = X[column].clip(lower_bound, upper_bound)
        else:
            for j in range(X.shape[1]):
                Q1 = np.percentile(X[:, j], 25)
                Q3 = np.percentile(X[:, j], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[:, j] = np.clip(X[:, j], lower_bound, upper_bound)
        return X
    
    def fit_transform(self, X, y=None):
        """
        Fit the preprocessor and transform the data.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        y : array-like, optional (default=None)
            Target variable (needed for supervised feature selection)
            
        Returns:
        --------
        X_transformed : array-like
            Transformed features
        """
        # Identify feature types
        self._identify_feature_types(X)
        
        # Initialize components
        self._initialize_components()
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = np.copy(X)
            
        # Handle outliers if specified
        if self.handle_outliers:
            X = self._handle_outliers(X)
            
        # Handle missing values
        if self.imputer:
            if isinstance(X, pd.DataFrame):
                X[self.numerical_features] = self.imputer.fit_transform(X[self.numerical_features])
            else:
                X = self.imputer.fit_transform(X)
                
        # Scale numerical features
        if self.num_scaler and len(self.numerical_features) > 0:
            if isinstance(X, pd.DataFrame):
                X[self.numerical_features] = self.num_scaler.fit_transform(X[self.numerical_features])
            else:
                X = self.num_scaler.fit_transform(X)
                
        # Encode categorical features
        if self.cat_encoder and len(self.categorical_features) > 0:
            if isinstance(X, pd.DataFrame):
                if self.categorical_encoding == 'label':
                    for col in self.categorical_features:
                        X[col] = self.cat_encoder.fit_transform(X[col])
                else:  # onehot
                    encoded_features = self.cat_encoder.fit_transform(X[self.categorical_features])
                    feature_names = self.cat_encoder.get_feature_names_out(self.categorical_features)
                    X = pd.concat([
                        X.drop(columns=self.categorical_features),
                        pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                    ], axis=1)
                    
        # Convert to numpy array for further processing
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        # Feature selection
        if self.feature_selector and y is not None:
            X = self.feature_selector.fit_transform(X, y)
            
        # Dimensionality reduction
        if self.pca:
            X = self.pca.fit_transform(X)
            
        return X
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
            
        Returns:
        --------
        X_transformed : array-like
            Transformed features
        """
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = np.copy(X)
            
        # Handle outliers if specified
        if self.handle_outliers:
            X = self._handle_outliers(X)
            
        # Handle missing values
        if self.imputer:
            if isinstance(X, pd.DataFrame):
                X[self.numerical_features] = self.imputer.transform(X[self.numerical_features])
            else:
                X = self.imputer.transform(X)
                
        # Scale numerical features
        if self.num_scaler and len(self.numerical_features) > 0:
            if isinstance(X, pd.DataFrame):
                X[self.numerical_features] = self.num_scaler.transform(X[self.numerical_features])
            else:
                X = self.num_scaler.transform(X)
                
        # Encode categorical features
        if self.cat_encoder and len(self.categorical_features) > 0:
            if isinstance(X, pd.DataFrame):
                if self.categorical_encoding == 'label':
                    for col in self.categorical_features:
                        X[col] = self.cat_encoder.transform(X[col])
                else:  # onehot
                    encoded_features = self.cat_encoder.transform(X[self.categorical_features])
                    feature_names = self.cat_encoder.get_feature_names_out(self.categorical_features)
                    X = pd.concat([
                        X.drop(columns=self.categorical_features),
                        pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                    ], axis=1)
                    
        # Convert to numpy array for further processing
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        # Feature selection
        if self.feature_selector:
            X = self.feature_selector.transform(X)
            
        # Dimensionality reduction
        if self.pca:
            X = self.pca.transform(X)
            
        return X

    def get_feature_names(self):
        """Get names of features after transformation."""
        if not hasattr(self, 'feature_names_'):
            warnings.warn("Feature names are not available. Transform data first.")
            return None
        return self.feature_names_ 