import pandas as pd
import os
import logging
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path):
    logger.debug(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df

def load_model(model_path):
    logger.debug(f"Loading model from {model_path}")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test):
    logger.debug(f"Evaluating model: {model}")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

def save_metrics(metrics, folder_path, file_name='metrics.json'):
    # Ensure folder exists before saving
    os.makedirs(folder_path, exist_ok=True)  # Create the 'report' folder if it doesn't exist
    logger.debug(f"Folder '{folder_path}' checked/created successfully.")
    file_path = os.path.join(folder_path, file_name)
    logger.debug(f"Saving metrics to {file_path}")
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)

def main():
    try:
        # Load the test data
        X_test_path = 'data/X_test.csv'
        y_test_path = 'data/y_test.csv'
        X_test = load_data(X_test_path)
        y_test = load_data(y_test_path)

        # List of model file paths
        model_paths = [
            'models\Gradient Boosting.pkl',
            'models\Histogram Gradient Boosting.pkl',
            'models\Logistic Regression.pkl',
            'models\Random Forest.pkl'
        ]
        
        # Dictionary to store metrics for each model
        all_metrics = {}

        # Evaluate each model
        for model_path in model_paths:
            model_name = os.path.basename(model_path).split('.')[0]  # Extract model name from file path
            model = load_model(model_path)
            logger.info(f"Evaluating {model_name}...")

            accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)

            # Store metrics for the current model
            all_metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }

        # Save all metrics to the 'report' folder
        report_folder = 'report'
        save_metrics(all_metrics, report_folder)

        logger.info("Model evaluation completed and metrics saved.")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
