import json
import math

import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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
    # X = check_and_replace_invalid_values(X)
    # y = check_and_replace_invalid_values(y)

    # Split the dataset into training (80%) and combined validation-test set (20%)
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Split the combined validation-test set into validation (10%) and test set (10%)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42, shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, output_columns, input_columns

def calculate_average(float_list):
    if len(float_list) == 0:
        return None  # Handle case where list is empty
    else:
        return sum(float_list) / len(float_list)
def get_similar_point(input_columns_values, input_file, target):
    input_df = pd.read_csv(input_file)
    similar_points = []

    for column, value in input_columns_values.items():
        min_diff = math.inf
        closest_point = None

        # Loop over each row to find the closest match for the current column
        for i in range(len(input_df)):
            known_value = input_df.iloc[i][column]
            difference = abs(known_value - value)

            # Update the closest point if the current row has a smaller difference
            if difference < min_diff:
                min_diff = difference
                closest_point = input_df.iloc[i][target]

        # Append the closest point for the current column to the result list
        similar_points.append(closest_point)

    return calculate_average(similar_points)
def scale_data(X_train, X_val, X_test, y_train, y_val, y_test):
    # Scale the input data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale the output data
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_y



def make_model(scaler_y, output_columns, input_columns, X_train, X_val, X_test, y_train, y_val, y_test):
    # Initialize the dictionary to store the best models and parameters
    svr_models = {}
    best_params = {}
    negative_count = 0

    # Define the parameter grid for SVR
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'gamma': ['scale', 'auto']
    }

    # Convert scaled arrays back to DataFrames for plotting
    X_test_df = pd.DataFrame(X_test, columns=input_columns)
    y_test_df = pd.DataFrame(y_test, columns=output_columns)

    # Loop through each output column and perform grid search
    for output in output_columns:
        print(f"Training model for {output}...")

        # Initialize the SVR model
        svr = SVR()

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train[:, output_columns.get_loc(output)])

        # Get the best estimator and parameters
        best_svr = grid_search.best_estimator_
        best_params[output] = grid_search.best_params_
        svr_models[output] = best_svr

        # Save the best model
        joblib.dump(best_svr, f'svr_model_{output}.pkl')
        save_params(f'svr_model_{output}', best_params[output])

        # Evaluate the model on scaled validation data
        y_pred_scaled = best_svr.predict(X_val)
        mse_scaled = mean_squared_error(y_val[:, output_columns.get_loc(output)], y_pred_scaled)
        print(f'Scaled MSE for {output} on validation set: {mse_scaled}')

        # Inverse transform the predictions and evaluate on original scale
        y_pred_scaled = best_svr.predict(X_val).reshape(-1, 1)
        y_pred_original = scaler_y.inverse_transform(
            np.hstack(
                [y_pred_scaled if i == output_columns.get_loc(output) else np.zeros_like(y_pred_scaled)
                 for i in range(len(output_columns))]
            )
        )[:, output_columns.get_loc(output)]


        mse_original = mean_squared_error(y_val[:, output_columns.get_loc(output)], y_pred_original)
        print(f'Original scale MSE for {output} on validation set: {mse_original}')


def predict_output(input_data_file, output_columns, scaler_y):
    """
    Predicts the output values for the given input dataset using trained SVR models and appends them to the input data.

    Parameters:
    - input_data_file: Path to the CSV file containing the input features.

    Returns:
    - DataFrame containing the input features and the predicted output values.
    """
    # Load the input data
    input_df = pd.read_csv(input_data_file)

    # Load the scaler for input data
    scaler_X = joblib.load('scaler_X.pkl')

    # Scale the input data
    scaled_input_data = scaler_X.transform(input_df)

    # Load the SVR models
    svr_models = {}
    for output in output_columns:
        svr_models[output] = joblib.load(f'svr_model_{output}.pkl')

    # Predict each output
    output_predictions = {}
    for output, model in svr_models.items():
        # Predict the output values using the corresponding SVR model
        output_predictions[output] = model.predict(scaled_input_data).reshape(-1, 1)

    # Inverse transform the predictions and store them in the DataFrame
    for output, predictions in output_predictions.items():
        y_pred_original = scaler_y.inverse_transform(
            np.hstack(
                [predictions if i == output_columns.get_loc(output) else np.zeros_like(predictions)
                 for i in range(len(output_columns))]
            )
        )[:, output_columns.get_loc(output)]
        input_df[output] = y_pred_original

    return input_df

def save_predictions(input_data_file, output_data_file, output_columns, scaler_y):
    """
    Predicts the output values for the given input dataset, appends them to the input data, and saves it to a new CSV file.

    Parameters:
    - input_data_file: Path to the CSV file containing the input features.
    - output_data_file: Path to the CSV file where the combined input and output data will be saved.
    """
    # Get the predictions
    combined_df = predict_output(input_data_file, output_columns, scaler_y)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_data_file, index=False)

# Example usage
def run():
    X_train, X_val, X_test, y_train, y_val, y_test, output_columns, input_columns = load_dataset('train_extended7.csv')
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_y = scale_data(X_train, X_val, X_test, y_train, y_val, y_test)
    make_model(scaler_y, output_columns, input_columns, X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled)

    # Predict and save output for a new input dataset
    input_data_file = 'submission.csv'  # This is your input file with 8 columns
    output_data_file = 'submission_predictions.csv'
    save_predictions(input_data_file, output_data_file, output_columns, scaler_y)

