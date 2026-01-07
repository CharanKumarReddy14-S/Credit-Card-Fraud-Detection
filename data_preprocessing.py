"""
Data Preprocessing Module for Credit Card Fraud Detection
Handles data loading, cleaning, feature engineering, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for fraud detection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """
        Load credit card transaction data
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with transaction data
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def explore_data(self, df):
        """
        Basic data exploration and statistics
        """
        print("\n=== Data Exploration ===")
        print(f"Total transactions: {len(df)}")
        print(f"\nClass distribution:")
        print(df['Class'].value_counts())
        print(f"\nFraud percentage: {(df['Class'].sum() / len(df)) * 100:.4f}%")
        print(f"\nMissing values:\n{df.isnull().sum().sum()}")
        print(f"\nData types:\n{df.dtypes.value_counts()}")
        
        return df
    
    def feature_engineering(self, df):
        """
        Create additional features from existing data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Time-based features
        if 'Time' in df.columns:
            # Convert to hours
            df['Hour'] = (df['Time'] / 3600) % 24
            df['Day'] = (df['Time'] / 86400).astype(int)
            
            # Time of day categories
            df['Time_of_Day'] = pd.cut(df['Hour'], 
                                       bins=[0, 6, 12, 18, 24],
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            df['Time_of_Day'] = df['Time_of_Day'].cat.codes
        
        # Amount-based features
        if 'Amount' in df.columns:
            df['Amount_log'] = np.log1p(df['Amount'])
            
            # Amount categories
            df['Amount_Category'] = pd.cut(df['Amount'],
                                          bins=[-np.inf, 10, 100, 500, np.inf],
                                          labels=[0, 1, 2, 3])
            df['Amount_Category'] = df['Amount_Category'].astype(int)
        
        print(f"\nFeatures after engineering: {df.shape[1]}")
        return df
    
    def prepare_features(self, df, target_col='Class'):
        """
        Separate features and target, handle scaling
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            X (features), y (target)
        """
        # Separate target
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        print(f"\nFeature shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def scale_features(self, X_train, X_test, X_val=None):
        """
        Apply StandardScaler to features
        
        Args:
            X_train: Training features
            X_test: Test features
            X_val: Validation features (optional)
            
        Returns:
            Scaled feature sets
        """
        print("\nScaling features...")
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            return X_train_scaled, X_test_scaled, X_val_scaled
        
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """
        Handle class imbalance using SMOTE or other techniques
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'smote' or 'none'
            
        Returns:
            Balanced X_train, y_train
        """
        print(f"\nHandling class imbalance using {method}...")
        print(f"Before balancing: {y_train.value_counts().to_dict()}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
            return X_train_balanced, y_train_balanced
        
        return X_train, y_train
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train and val
        val_proportion = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_proportion, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """
        Save scaler and feature columns for later use
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"\nPreprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """
        Load saved preprocessor
        """
        preprocessor_data = joblib.load(filepath)
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        print(f"\nPreprocessor loaded from {filepath}")
        
        return self


def preprocess_pipeline(filepath, save_processed=True):
    """
    Complete preprocessing pipeline
    
    Args:
        filepath: Path to raw data CSV
        save_processed: Whether to save processed data
        
    Returns:
        Preprocessed and split data
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=42)
    
    # Load and explore
    df = preprocessor.load_data(filepath)
    df = preprocessor.explore_data(df)
    
    # Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # Prepare features
    X, y = preprocessor.prepare_features(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, X_val_scaled = preprocessor.scale_features(
        X_train, X_test, X_val
    )
    
    # Handle imbalance on training set only
    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(
        X_train_scaled, y_train, method='smote'
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Optionally save processed data
    if save_processed:
        os.makedirs('data/processed', exist_ok=True)
        
        # Save splits
        pd.concat([X_train_balanced, y_train_balanced], axis=1).to_csv(
            'data/processed/train.csv', index=False
        )
        pd.concat([X_val_scaled, y_val], axis=1).to_csv(
            'data/processed/val.csv', index=False
        )
        pd.concat([X_test_scaled, y_test], axis=1).to_csv(
            'data/processed/test.csv', index=False
        )
        
        print("\nProcessed data saved to data/processed/")
    
    return X_train_balanced, X_val_scaled, X_test_scaled, y_train_balanced, y_val, y_test


if __name__ == "__main__":
    # Example usage
    data_path = "data/creditcard.csv"
    
    if os.path.exists(data_path):
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(data_path)
        print("\n✅ Preprocessing completed successfully!")
    else:
        print(f"❌ Data file not found at {data_path}")
        print("Please download the dataset and place it in the data/ directory")