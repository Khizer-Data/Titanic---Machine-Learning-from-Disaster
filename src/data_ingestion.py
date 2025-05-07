import pandas as pd
import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Ensure that the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def download_data():
    """
    Download the Titanic dataset directly from Kaggle.
    """
    try:
        api = KaggleApi()
        api.authenticate()

        # Download the Titanic dataset
        download_path = 'data/'
        api.competition_download_files('titanic', path=download_path)
        logger.debug("Data downloaded successfully from Kaggle.")

        # Extract the ZIP file
        zip_file_path = os.path.join(download_path, 'titanic.zip')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
            logger.debug("ZIP file extracted successfully.")

        # Delete the ZIP file after extraction
        os.remove(zip_file_path)
        logger.debug(f"Deleted ZIP file: {zip_file_path}")

    except Exception as e:
        logger.error(f"Error downloading or extracting data from Kaggle: {e}")
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully from {data_url}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise

def main():
    """
    Main function to load data.
    """
    try:
        # Download Titanic data
        download_data()

        # Example usage after downloading and extracting
        train_data_path = 'data/train.csv'
        test_data_path = 'data/test.csv'

        # Load the train and test data
        train_df = load_data(train_data_path)
        test_df = load_data(test_data_path)

        # Print the first few rows
        print(train_df.head())
        print(test_df.head())

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

if __name__ == '__main__':
    main()
