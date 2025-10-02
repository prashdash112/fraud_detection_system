from src.preprocessing.preprocessing import *

def run():
    INPUT_FILE = "./src/data/creditcard.csv" 
    OUTPUT_FILE = "./src/data/creditcard_preprocessed.csv"
        
    processed_df = preprocess_pipeline(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        missing_strategy='mean',
        scaling_method='standard'
        )

    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"\nFirst few rows:\n{processed_df.head()}")
    print(f"\nData statistics:\n{processed_df.describe()}")


if __name__ == "__main__":
    run() 

