import os
import logging
import pickle
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier

# Logging setup
def setup_logger():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('model_training')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(log_dir, 'model_training.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

# Load YAML parameters
def load_params(path='params.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# Label encoding
def label_encoding(df, columns, logger):
    logger.debug("Label encoding started")
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
        else:
            logger.warning(f"Column '{col}' not found for label encoding.")
    logger.debug("Label encoding completed")
    return df

# Standardization
def standardization(df, columns, logger):
    logger.debug("Standardization started")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.debug("Standardization completed")
    return df

# Data splitting
def split_data(df, target, test_size, random_state, logger):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Data split: {X_train.shape}, {X_test.shape}")
    return X_train, X_test, y_train, y_test

# Save model
def save_model(model, name, model_dir, logger):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model: {path}")

# Save CSV
def save_csv(df, path, logger):
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")

# Train models
def train_models(X_train, y_train, X_test, y_test, params, model_dir, logger):
    models = {
        "logistic_regression": LogisticRegression(**params["logistic_regression"]),
        "gradient_boosting": GradientBoostingClassifier(**params["gradient_boosting"]),
        "histogram_gradient_boosting": HistGradientBoostingClassifier(**params["histogram_gradient_boosting"]),
        "random_forest": RandomForestClassifier(**params["random_forest"])
    }

    for name, model in models.items():
        logger.info(f"Training {name}")
        model.fit(X_train, y_train)
        _ = model.predict(X_test)
        save_model(model, name, model_dir, logger)
        logger.info(f"{name} completed\n")

# Main
def main():
    logger = setup_logger()
    logger.info("Starting training pipeline")

    params = load_params()
    training_params = params["model_training"]
    paths = params["data_paths"]

    if not os.path.exists(paths["input_csv"]):
        logger.error(f"File not found: {paths['input_csv']}")
        return

    df = pd.read_csv(paths["input_csv"])
    logger.info("CSV loaded")

    if df['Survived'].isnull().any():
        logger.warning("Missing target values found. Dropping rows.")
        df = df.dropna(subset=['Survived'])

    df = label_encoding(df, ['Sex', 'Embarked', 'Title'], logger)
    df = standardization(df, ['Age', 'FamilySize', 'FarePerPerson', 'Age*Class'], logger)

    df = df.dropna()  # Final check
    save_csv(df, paths["output_csv"], logger)

    X_train, X_test, y_train, y_test = split_data(
        df, 'Survived', training_params["test_size"], training_params["random_state"], logger
    )

    save_csv(X_test, paths["x_test_csv"], logger)
    save_csv(y_test, paths["y_test_csv"], logger)

    train_models(X_train, y_train, X_test, y_test, training_params, paths["model_dir"], logger)

    logger.info("Training pipeline completed")

if __name__ == "__main__":
    main()
