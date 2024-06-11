import json
import optuna
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from SVR import _load_dataset, load_dataset

X_train, X_test, y_train, y_test, output_columns, input_columns = load_dataset('initial_data.csv')
X, y = _load_dataset('initial_data.csv')

def objective(trial, X_train, X_test, y_train, y_test):
    # Define hyperparameter search space
    kernels = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
    epsilons = trial.suggest_float('epsilon', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
    C = trial.suggest_float('C', 1e-5, 1e-1, log=True)
    degree = trial.suggest_int('degree', 2, 5)

    # Create the SVR model
    reg = SVR(
        kernel=kernels,
        epsilon=epsilons,
        gamma=gamma,
        C=C,
        degree=degree
    )

    # Fit the model
    reg.fit(X_train, y_train)
    # Predict on the test set
    y_pred = reg.predict(X_test)
    # Calculate the MSE on the test set
    mse = mean_squared_error(y_test, y_pred)

    return mse

for output_column in output_columns:
    X_train, X_test, y_train, y_test = train_test_split(X, y[output_column], test_size=0.2, random_state=42, shuffle=True)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=200)

    # Print the best hyperparameters and MSE
    print(f"Best hyperparameters for {output_column}: {study.best_params}")
    print(f"Best MSE for {output_column}: {study.best_value}")

    with open(f"_optunaParamOutput_{output_column}.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
