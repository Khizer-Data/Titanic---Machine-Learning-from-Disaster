import pandas as pd
import os
import logging
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dvclive import Live
import yaml




def load_data(file_path):
    logger.debug(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df

def load_model(model_path, logger):
    """Load a trained model from disk"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Loading model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def setup_logger():
    """Set up and return logger"""
    logger = logging.getLogger('model_evaluation')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join('logs', 'model_evaluation.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_test = y_test.values.ravel()  # Convert to 1D array if needed
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1, roc_auc

def save_metrics(metrics, report_folder):
    """Save metrics to JSON file"""
    os.makedirs(report_folder, exist_ok=True)
    metrics_file = os.path.join(report_folder, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_params(path='params.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    logger = setup_logger()
    
    try:
        # Load parameters
        params = load_params()
        
        # Load test data
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        
        # Define model paths with underscores instead of spaces
        model_paths = {
            'logistic_regression': 'models/logistic_regression.pkl',
            'gradient_boosting': 'models/gradient_boosting.pkl',
            'histogram_gradient_boosting': 'models/histogram_gradient_boosting.pkl',
            'random_forest': 'models/random_forest.pkl'
        }
        
        all_metrics = {}
        # Experiment tracking using DVC Live
        with Live(save_dvc_exp=True) as live:
            for model_name, model_path in model_paths.items():
                model = load_model(model_path, logger)
                logger.info(f"Evaluating {model_name}...")

                accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)

                # Store metrics for the current model
                model_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }
                all_metrics[model_name] = model_metrics

                # Log metrics individually for each model
                for metric_name, value in model_metrics.items():
                    live.log_metric(f"{model_name}_{metric_name}", value)

            # Log parameters
            for param_name, param_value in params.items():
                live.log_param(param_name, param_value)

        # Save all metrics to the 'report' folder
        save_metrics(all_metrics, 'report')
        logger.info("Model evaluation completed and metrics saved.")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
