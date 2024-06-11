import json
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from SVR import save_predictions
def save_params(name, params):
    with open(f'{name}_bestParamaters.json', 'w') as fh:
        fh.write(json.dumps(params, indent=4))

def check_and_replace_invalid_values(df):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaN values with the mean of the column
    df = df.fillna(df.mean())
    return df

def load_dataset(dataset):
    # Load the dataset
    df = pd.read_csv(dataset)

    # Assuming your CSV columns are named appropriately
    input_columns = df.columns[1:9]
    output_columns = df.columns[9:10]

    # Separate input (X) and output (y)
    X = df[input_columns]
    y = df[output_columns]

    # Check and replace invalid values in X and y
    #X = check_and_replace_invalid_values(X)
    #y = check_and_replace_invalid_values(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, output_columns, input_columns

def scale_data(X_train, X_test, y_train, y_test):
    # Scale the input data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale the output data
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y

def make_model(scaler_y, output_columns, input_columns, X_train, X_test, y_train, y_test):
    # Initialize the dictionary to store the best models and parameters
    xgb_models = {}
    best_params = {}

    # Define the parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    # Convert scaled arrays back to DataFrames for plotting
    X_test_df = pd.DataFrame(X_test, columns=input_columns)
    y_test_df = pd.DataFrame(y_test, columns=output_columns)

    # Loop through each output column and perform grid search
    for output in output_columns:
        print(f"Training model for {output}...")

        # Initialize the XGBoost model
        xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train[:, output_columns.get_loc(output)])

        # Get the best estimator and parameters
        best_xgb = grid_search.best_estimator_
        best_params[output] = grid_search.best_params_
        xgb_models[output] = best_xgb

        # Save the best model
        joblib.dump(best_xgb, f'xgb_model_{output}.pkl')
        save_params(f'xgb_model_{output}', best_params[output])

        # Evaluate the model on scaled data
        y_pred_scaled = best_xgb.predict(X_test)
        mse_scaled = mean_squared_error(y_test[:, output_columns.get_loc(output)], y_pred_scaled)
        print(f'Scaled MSE for {output}: {mse_scaled}')

        # Inverse transform the predictions and evaluate on original scale
        y_pred_scaled = best_xgb.predict(X_test).reshape(-1, 1)
        y_pred_original = scaler_y.inverse_transform(
            np.hstack(
                [y_pred_scaled if i == output_columns.get_loc(output) else np.zeros_like(y_pred_scaled)
                 for i in range(len(output_columns))]
            )
        )[:, output_columns.get_loc(output)]
        mse_original = mean_squared_error(y_test[:, output_columns.get_loc(output)], y_pred_original)
        print(f'Original scale MSE for {output}: {mse_original}')

def check_nan(dataset):
    # Load the dataset
    df = pd.read_csv(dataset)

    # Check for NaNs in the dataset
    print("Checking for NaNs in the dataset:")
    print(df.isnull().sum())

    # Drop rows with NaN values
    df = df.dropna()

    # Save cleaned dataset (optional, for debugging purposes)
    df.to_csv('cleaned_standardized_data.csv', index=False)

def cross_val(model, X, y, n_splits=5):
    """
    Perform cross-validation for regression models.

    Parameters:
    - model: The regression model to be evaluated.
    - X: Feature dataset.
    - y: Target dataset.
    - n_splits: Number of folds for cross-validation.

    Returns:
    - List of MSE scores across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit the model
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_val_pred = model.predict(X_val)

        # Calculate MSE
        mse = mean_squared_error(y_val, y_val_pred)
        mse_scores.append(mse)
        print(f'Fold {fold + 1}, MSE on Validation Set: {mse}')

    average_mse = np.mean(mse_scores)
    print(f'\nAverage MSE across {n_splits} folds: {average_mse}')
    return mse_scores

def predict_output(input_data_file, output_columns, scaler_y):
    """
    Predicts the output values for the given input dataset using trained XGBoost models.

    Parameters:
    - input_data_file: Path to the CSV file containing the input features.

    Returns:
    - DataFrame containing the predicted output values.
    """
    # Load the input data
    input_df = pd.read_csv(input_data_file)

    # Load the scaler for input data
    scaler_X = joblib.load('scaler_X.pkl')

    # Scale the input data
    scaled_input_data = scaler_X.transform(input_df)

    # Load the XGBoost models
    xgb_models = {}
    for output in output_columns:
        xgb_models[output] = joblib.load(f'xgb_model_{output}.pkl')

    # Predict each output
    output_predictions = {}
    for output, model in xgb_models.items():
        # Predict the output values using the corresponding XGBoost model
        output_predictions[output] = model.predict(scaled_input_data).reshape(-1, 1)

    # Inverse transform the predictions and store them in a DataFrame
    output_predictions_df = pd.DataFrame()
    for output, predictions in output_predictions.items():
        y_pred_original = scaler_y.inverse_transform(
            np.hstack(
                [predictions if i == output_columns.get_loc(output) else np.zeros_like(predictions)
                 for i in range(len(output_columns))]
            )
        )[:, output_columns.get_loc(output)]
        output_predictions_df[output] = y_pred_original

    return output_predictions_df

# Example usage
def run():
    X_train, X_test, y_train, y_test, output_columns, input_columns = load_dataset('train_extended7_CO2.csv')
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y = scale_data(X_train, X_test, y_train, y_test)
    make_model(scaler_y, output_columns, input_columns, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)

    # Predict and save output for a new input dataset
    input_data_file = 'submission.csv'  # This is your input file with 8 columns
    output_data_file = 'submission_predictions.csv'
    save_predictions(input_data_file, output_data_file, output_columns, scaler_y)

run()
