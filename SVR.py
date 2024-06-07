import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def save_params(name, params):
    with open(f'{name}_bestParamaters.json', 'w') as fh:
        fh.write(json.dumps(params, indent=4))

# Load the dataset
df = pd.read_csv('initial_data.csv')

# Assuming your CSV columns are named appropriately
input_columns = df.columns[:8]
output_columns = df.columns[8:13]

# Separate input (X) and output (y)
X = df[input_columns]
y = df[output_columns]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the output data
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.5],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

# Initialize the dictionary to store the best models
svr_models = {}

# Perform grid search and train the model for each output column
for output in output_columns:
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_scaled[:, output_columns.get_loc(output)])

    # Best hyperparameters
    print(f"Best hyperparameters for {output}: {grid_search.best_params_}")

    # Train the final model using the best hyperparameters
    best_svr = grid_search.best_estimator_
    best_svr.fit(X_train_scaled, y_train_scaled[:, output_columns.get_loc(output)])
    svr_models[output] = best_svr

    joblib.dump(best_svr, f'svr_model_{output}.pkl')
    save_params(output, grid_search.best_params_)

# Evaluate the models on scaled data
scaled_mse = {}
for output in output_columns:
    y_pred_scaled = svr_models[output].predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled[:, output_columns.get_loc(output)], y_pred_scaled)
    scaled_mse[output] = mse
    print(f'Scaled MSE for {output}: {mse}')

# Inverse transform the predictions and evaluate on original scale
original_mse = {}
for output in output_columns:
    y_pred_scaled = svr_models[output].predict(X_test_scaled).reshape(-1, 1)
    y_pred_original = scaler_y.inverse_transform(
        np.hstack(
            [y_pred_scaled if i == output_columns.get_loc(output) else np.zeros_like(y_pred_scaled)
             for i in range(len(output_columns))]
        )
    )[:, output_columns.get_loc(output)]
    mse = mean_squared_error(y_test[output], y_pred_original)
    original_mse[output] = mse
    print(f'Original scale MSE for {output}: {mse}')

