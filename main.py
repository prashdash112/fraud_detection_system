from src.preprocessing.preprocessing import *
from src.preprocessing.handle_imbalance import *

def run():
    ####### data loading and preprocessing #######
    # INPUT_FILE = "./src/data/creditcard.csv" 
    # OUTPUT_FILE = "./src/data/creditcard_preprocessed.csv"
        
    # processed_df = preprocess_pipeline(
    #     input_path=INPUT_FILE,
    #     output_path=OUTPUT_FILE,
    #     missing_strategy='mean',
    #     scaling_method='standard'
    #     )

    # print(f"\nProcessed data shape: {processed_df.shape}")
    # print(f"\nFirst few rows:\n{processed_df.head()}")
    # print(f"\nData statistics:\n{processed_df.describe()}")

    #########################################################
    # Configuration - Balancing data and train test split 
    INPUT_preprocessed_FILE = "./src/data/creditcard_preprocessed.csv"
    TRAIN_balanced_OUTPUT = "./src/data/creditcard_preprocessed_balanced_train.csv"
    TEST_balanced_OUTPUT = "./src/data/creditcard_preprocessed_balanced_test.csv"
    
    print("\n" + "="*60)
    print("FRAUD DETECTION: CLASS IMBALANCE HANDLING")
    print("="*60)
    
    # Load data
    X, y = load_imbalanced_data(INPUT_preprocessed_FILE)
    show_distribution(y, "Original Data")
    
    # Split and balance
    X_train, X_test, y_train, y_test = split_and_balance(X, y, test_size=0.2)
    
    # Save processed data
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    save_balanced_data(X_train, y_train, TRAIN_balanced_OUTPUT, feature_cols)
    save_balanced_data(X_test, y_test, TEST_balanced_OUTPUT, feature_cols)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training samples: {len(y_train):,}")
    print(f"Test samples: {len(y_test):,}")
    print(f"\n✓ Ready for model training!")
    print("="*60)


if __name__ == "__main__":
    run() 

