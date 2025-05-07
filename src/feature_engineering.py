import pandas as pd
import os 
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def feature_engineering(df):
    try:
    
        for _ in [df]:
        # Extract title from the 'Name' column
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            logger.debug("Extracted Title from Name")
            df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') 
            logger.debug("Rare Title replaced")  
            df['Title'] = df['Title'].replace('Mlle', 'Miss')
            logger.debug("Mlle replaced with Miss")
            df['Title'] = df['Title'].replace('Ms', 'Miss')
            logger.debug("Ms replaced with Miss")
            df['Title'] = df['Title'].replace('Mme', 'Mrs')
            logger.debug("Mme replaced with Mrs")

            # create a new feature 'FamilySize' by adding 'SibSp' and 'Parch'
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            logger.debug(" New Feature FamilySize created")

            # create a new feature 'IsAlone' based on 'FamilySize'
            df['IsAlone'] = 0
            df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
            logger.debug(" New Feature IsAlone created")
            # create a new feature 'FarePerPerson' by dividing 'Fare' by 'FamilySize'
            df['FarePerPerson'] = df['Fare'] / (df['FamilySize'] + 1)
            logger.debug(" New Feature FarePerPerson created")

            # create a new feature 'Ageband' based on 'Age'
            df['Ageband'] = pd.cut(df['Age'], 5)
            logger.debug(" New Feature Ageband created")

            # create a new feature 'Age*Class'
            df['Age*Class'] = df['Age'] * df['Pclass']
            logger.debug(" New Feature Age*Class created")

            # create a fare band
            df['FareBand'] = pd.qcut(df['Fare'], 4)
            logger.debug(" New Feature FareBand created")

            # Has Cabin
            df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if isinstance(x, float) else 1)
            logger.debug(" New Feature HasCabin created")

            # Drop unnecessary columns
            df = df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin','PassengerId','FareBand','Ageband'], axis=1)
            logger.debug("Unnecessary columns dropped")
        return df
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise

def main():
    df = pd.read_csv('data/preprocessed_data.csv')
    logger.info("Data loaded successfully")
    df = feature_engineering(df)
    df.to_csv('data/feature_engineering.csv', index=False)
    logger.info("Data saved successfully")
    logger.info("Feature engineering completed")

if __name__ == '__main__':
    main()