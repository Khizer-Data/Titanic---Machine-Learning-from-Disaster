import os
import pandas as pd
import logging

# Ensure that the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_data(df_combined):
    """
    Preprocess the combined Titanic dataset.
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        df = df_combined.copy()
        
        # Impute Age based on Pclass
        for pclass in df['Pclass'].unique():
            median_age = df[df['Pclass'] == pclass]['Age'].median()
            mask = (df['Pclass'] == pclass) & (df['Age'].isnull())
            df.loc[mask, 'Age'] = median_age

        # Impute Embarked with most frequent
        if df['Embarked'].isnull().sum() > 0:
            most_common_embarked = df['Embarked'].mode()[0]
            df.loc[df['Embarked'].isnull(), 'Embarked'] = most_common_embarked

        # Impute Fare based on Pclass
        for pclass in df['Pclass'].unique():
            median_fare = df[df['Pclass'] == pclass]['Fare'].median()
            mask = (df['Pclass'] == pclass) & (df['Fare'].isnull())
            df.loc[mask, 'Fare'] = median_fare

        # Log missing values
        logger.debug("Missing values after imputation:\n" + str(df[['Age', 'Embarked', 'Fare']].isnull().sum()))

        # Log duplicates
        logger.debug(f"Number of duplicate rows: {df.duplicated().sum()}")

        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def main():
    try:
        logger.info("Starting Titanic data preprocessing...")

        # Load data
        logger.info("Loading training and test datasets.")
        df_train = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')
        logger.debug(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

        # Combine data
        logger.info("Combining train and test datasets.")
        df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
        logger.debug(f"Combined data shape: {df_combined.shape}")

        # Preprocess combined data
        logger.info("Preprocessing combined dataset.")
        df_combined = preprocess_data(df_combined)

        # Save the preprocessed combined data
        logger.info("Saving combined preprocessed dataset to 'data/preprocessed_data.csv'.")
        df_combined.to_csv('data/preprocessed_data.csv', index=False)

        logger.info("Preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
