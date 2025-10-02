import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data."""
    logger.info(f"Loading data from {filepath}") 
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df 


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric types using numpy."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info("Converted columns to numeric")
    return df


def handle_missing(df: pd.DataFrame, strategy='drop') -> pd.DataFrame:
    """Handle missing values using sklearn SimpleImputer.""" 
    missing_count = np.isnan(df.values).sum() 
    logger.info(f"Found {missing_count} missing values")
    
    if missing_count == 0:
        return df
    
    if strategy == 'drop':
        df = df.dropna() 
    else:
        # Use sklearn's SimpleImputer
        strategy_map = {
            'mean': 'mean',
            'median': 'median',
            'zero': 'constant'
        }
        
        imputer = SimpleImputer(
            strategy=strategy_map.get(strategy, 'mean'),
            fill_value=0 if strategy == 'zero' else None
        )
        
        df_imputed = imputer.fit_transform(df)
        df = pd.DataFrame(df_imputed, columns=df.columns)
    
    logger.info(f"Handled missing values using '{strategy}'")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows using numpy."""
    before = len(df)
    
    arr = df.values
    _, unique_indices = np.unique(arr, axis=0, return_index=True)
    df = df.iloc[sorted(unique_indices)]
    
    removed = before - len(df)
    logger.info(f"Removed {removed} duplicates")
    
    return df.reset_index(drop=True) 


def scale_features(df: pd.DataFrame, columns: list, method='standard') -> pd.DataFrame:
    """
    Scale features using sklearn scalers.
    
    Args:
        df: DataFrame
        columns: List of columns to scale
        method: 'minmax', 'standard', or 'robust'
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    df[columns] = scaler.fit_transform(df[columns])
    logger.info(f"Scaled {len(columns)} columns using {method} scaler")
    
    return df



def validate_class_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Class column is binary (0 or 1) using numpy."""
    df['Class'] = df['Class'].astype(int)
    unique_vals = np.unique(df['Class'].values)
    
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Invalid Class values: {unique_vals}")
    
    return df 



def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed data to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics using numpy."""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': int(np.isnan(df.values).sum()),
        'duplicate_rows': int(len(df) - len(np.unique(df.values, axis=0))),
        'class_distribution': dict(zip(*np.unique(df['Class'].values, return_counts=True)))
    }
    return summary


def preprocess_pipeline(input_path: str, output_path: str, 
                       missing_strategy='drop', 
                       scaling_method='minmax'
                       ) -> pd.DataFrame:
    """
    Main preprocessing pipeline using sklearn and numpy.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save processed CSV
        missing_strategy: 'drop', 'mean', 'median', or 'zero'
        scaling_method: 'minmax', 'standard', or 'robust'
    
    Returns:
        Processed DataFrame
    """
    # Load and validate
    df = load_data(input_path)
    df = convert_to_numeric(df)
    
    # Clean data
    df = handle_missing(df, strategy=missing_strategy)
    df = remove_duplicates(df)
    
    # Scale features
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    df = scale_features(df, feature_cols, method=scaling_method)
    
    df = validate_class_labels(df) 
    
    # Summary
    summary = get_data_summary(df)
    logger.info(f"Processing complete: {summary}")
    
    # Save
    save_data(df, output_path)
    
    return df
