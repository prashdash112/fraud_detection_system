import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_imbalanced_data(filepath: str) -> tuple:
    """Load data and separate features from target."""
    df = pd.read_csv(filepath)
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Original distribution: {dict(Counter(y))}")
    
    return X, y


def show_distribution(y: np.ndarray, label: str = "") -> None:
    """Display class distribution with percentages."""
    counter = Counter(y)
    total = len(y)
    
    print(f"\n{label}:")
    for cls, count in counter.items():
        pct = (count / total) * 100
        print(f"  Class {cls}: {count:,} samples ({pct:.2f}%)")


def balance_with_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Balance using SMOTE (Synthetic Minority Over-sampling Technique).
    
    How it works:
    1. Finds k nearest neighbors of each fraud sample
    2. Creates synthetic fraud samples between them
    3. Balances classes without just copying existing samples
    """
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info("✓ Applied SMOTE balancing")
    show_distribution(y_balanced, "After SMOTE")
    
    return X_balanced, y_balanced


def split_and_balance(X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2):
    """
    Split data then balance ONLY the training set.
    
    Important: Balance AFTER splitting to avoid data leakage!
    """
    # Step 1: Split first (stratified to maintain fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    logger.info("\n=== After Train-Test Split ===")
    show_distribution(y_train, "Training set")
    show_distribution(y_test, "Test set")
    
    # Step 2: Balance ONLY training set
    X_train_balanced, y_train_balanced = balance_with_smote(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test


def save_balanced_data(X: np.ndarray, y: np.ndarray, filepath: str, 
             feature_names: list = None) -> None:
    """Save processed data to CSV."""
    if feature_names is None:
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y
    df.to_csv(filepath, index=False)
    
    logger.info(f"✓ Saved to {filepath}")

