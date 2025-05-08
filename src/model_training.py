import pandas as pd
import os
import logging
import pickle  # Add this import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Label encoding
def label_encoding(df, columns):
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
def standardization(df, columns):
    logger.debug("Standardization started")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.debug("Standardization completed")
    return df

# Data splitting
def data_splitting(df, target_column, test_size, random_state):
    logger.debug("Data splitting started")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.debug("Data splitting completed")
    return X_train, X_test, y_train, y_test

# Model functions
def logistic_regression(X_train, y_train, X_test, y_test):
    logger.debug("Logistic Regression started")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.debug("Logistic Regression completed")
    return model, y_pred

def gradient_boosting(X_train, y_train, X_test, y_test):
    logger.debug("Gradient Boosting started")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.debug("Gradient Boosting completed")
    return model, y_pred

def histogram_gradient_boosting(X_train, y_train, X_test, y_test):
    logger.debug("Histogram-based Gradient Boosting started")
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.debug("Histogram-based Gradient Boosting completed")
    return model, y_pred

def random_forest(X_train, y_train, X_test, y_test):
    logger.debug("Random Forest started")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.debug("Random Forest completed")
    return model, y_pred

# Save model
def save_model(model, model_name):
    logger.debug(f"Saving {model_name} model")
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)  # Ensure 'models' directory exists
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    logger.debug(f"{model_name} model saved at {model_path}")

# Main pipeline
def main():
    try:
        csv_path = os.path.join("data", "feature_engineering.csv")
        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        logger.debug("Data loaded successfully")

        # Check and handle missing values in target variable
        if df['Survived'].isnull().any():
            logger.warning(f"Found {df['Survived'].isnull().sum()} missing values in target variable")
            # Remove rows with missing target values
            df = df.dropna(subset=['Survived'])
            logger.info("Removed rows with missing target values")

        # Proceed with feature engineering
        df = label_encoding(df, ['Sex', 'Embarked', 'Title'])
        df = standardization(df, ['Age', 'FamilySize', 'FarePerPerson', 'Age*Class'])

        # Ensure no NaN values in features
        features = ['Sex', 'Embarked', 'Title', 'Age', 'FamilySize', 'FarePerPerson', 'Age*Class']
        if df[features].isnull().any().any():
            logger.warning("Found missing values in features. Dropping rows with missing values.")
            df = df.dropna(subset=features)
            logger.info(f"Shape after dropping missing values: {df.shape}")
        # save last df
        df.to_csv('data/model_training.csv', index=False)
        logger.info("Data saved successfully")
        X_train, X_test, y_train, y_test = data_splitting(df, 'Survived', test_size=0.21, random_state=42)
        # save test data
        X_test.to_csv('data/X_test.csv', index=False)
        logger.info("X_test saved successfully")
        y_test.to_csv('data/y_test.csv', index=False)
        logger.info("y_test saved successfully")
        logger.info("Data splitting completed")
        # Train and log models
        models = {
            "Logistic Regression": logistic_regression,
            "Gradient Boosting": gradient_boosting,
            "Histogram Gradient Boosting": histogram_gradient_boosting,
            "Random Forest": random_forest
        }

        for name, func in models.items():
            logger.info(f"Training {name}...")
            model, y_pred = func(X_train, y_train, X_test, y_test)
            logger.info(f"{name} training completed.")
            save_model(model, name)  # Save each model after training

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
