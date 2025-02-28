Creating a complete Python program for a project like "Eco-Insights" involves several components, including data collection, preprocessing, model training, predictions, and evaluation. Below is a simplified version that provides a fundamental structure for such a project. This includes the structure for handling data, building a basic machine learning model, and making predictions.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Loads data from a CSV file.

    :param file_path: str: The path to the CSV file.
    :return: DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info('Data loaded successfully.')
        return data
    except FileNotFoundError:
        logging.error('File not found.')
    except pd.errors.EmptyDataError:
        logging.error('No data in file.')
    except Exception as e:
        logging.error(f'An error occurred: {e}')

def preprocess_data(data):
    """
    Preprocesses the data by handling missing values and encoding categorical features.

    :param data: DataFrame: The input data.
    :return: DataFrame: Preprocessed data.
    """
    try:
        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # Encoding categorical features if needed
        # Assume 'Category' is a categorical column in the dataset
        if 'Category' in data.columns:
            data = pd.get_dummies(data, columns=['Category'])

        logging.info('Data preprocessed successfully.')
        return data
    except Exception as e:
        logging.error(f'An error occurred during preprocessing: {e}')

def train_model(X, y):
    """
    Trains a Random Forest model on the provided data.

    :param X: DataFrame: Features for training the model.
    :param y: Series: Target variable.
    :return: Trained RandomForestRegressor model.
    """
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        logging.info('Model trained successfully.')
        return model
    except Exception as e:
        logging.error(f'An error occurred during model training: {e}')

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using Mean Squared Error.

    :param model: Trained model.
    :param X_test: DataFrame: Test data features.
    :param y_test: Series: Test data target variable.
    :return: float: The mean squared error of the model.
    """
    try:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f'Model evaluation completed. MSE: {mse:.2f}')
        return mse
    except Exception as e:
        logging.error(f'An error occurred during model evaluation: {e}')

def main():
    # Load data
    file_path = 'urban_energy_data.csv'  # Placeholder path
    data = load_data(file_path)

    if data is not None:
        # Preprocess data
        data = preprocess_data(data)

        # Assuming the target column is 'EnergyConsumption'
        X = data.drop(columns=['EnergyConsumption'])
        y = data['EnergyConsumption']

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        if model is not None:
            # Evaluate the model
            mse = evaluate_model(model, X_test, y_test)
            logging.info(f'The Mean Squared Error of the model is: {mse:.2f}')
        else:
            logging.error('Model training failed, unable to evaluate.')
    else:
        logging.error('Data loading failed, exiting program.')

if __name__ == '__main__':
    main()
```

### Notes:
1. **Data Input**: The program assumes there's a dataset available named `urban_energy_data.csv`. This file should contain features useful for predicting energy consumption, such as average temperature, population density, etc.

2. **Preprocessing** includes basic imputation of missing values and encoding of categorical features. You might need to adjust this based on exact dataset characteristics.

3. **Model**: The program uses a `RandomForestRegressor` as a simple model example. Depending on the dataset's complexity and size, adjusting model parameters, or choosing a different model type might yield better results.

4. **Error Handling**: Basic error handling is implemented using try-except blocks and logging for debugging purposes.

5. **Scalability**: In a real-world scenario, you may need to consider scaling your data, hyperparameter tuning, and potentially integrating more advanced machine learning techniques or models.