import json
import joblib
import argparse
import logging
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run(args):
    '''
    Function that reads csv file with train-test dataset, splits the data
    in train-test datasets, and outputs the model

    input:
      args: parser object with train_data and rf_params attributes
    output:
      None: the function saves the model
    '''
    try:
        # Read train-test dataset
        logger.debug(f"Reading csv file with train-test data")
        dataset = pd.read_csv(args.train_data,index_col=0)

        # Split dataset
        logger.debug(f"Splitting into train-test datasets")
        X = dataset
        y = X.pop("flow7")
        # Shuffle = false to avoid problems with autocorrelation when testing the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size = 0.2)
        print(f"There are {len(X_train)//53} years in training set.")
        print(f"There are {len(X_test)//53} years in test set.")

        # Get dictionary of params for rf model
        logger.debug(f"Reading RF params from json file")
        with open(args.rf_params) as fp:
            rf_params = json.load(fp)


        # Define and train random forest model
        logger.debug(f"Defining and training model")
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)

        # Basic performance metrics
        logger.debug(f"Computing basic metrics of RF model")
        y_pred = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"The RF model has coefficient of determination {r2}")
        print(f"The RF model has mean absolute error {mae}\n")

        # Compare to a multilinear regression
        logger.debug(f"Computing basic metrics of linear model")
        lin_model = LinearRegression().fit(X_train, y_train)
        y_lin = lin_model.predict(X_test)
        r2_lin = r2_score(y_test, y_lin)
        mae_lin = mean_absolute_error(y_test, y_lin)
        print(f"A linear model has coefficient of determination {r2_lin}")
        print(f"A linear model has mean absolute error {mae_lin}\n")

        # Print message if there is evidence of overfitting
        y_pred_train = rf_model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        if (r2_train-r2)/r2 > .2:
            print("The model might be overfitting.")
            print(f"Coef of determination in training set is {r2_train} > {r2}")

        # Save RF model
        joblib.dump(rf_model, os.path.join("..","Models","rf_model.joblib"))

    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data from NetCDF to get train-test dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Path to csv file with features and predictand",
        required=True,
    )

    parser.add_argument(
        "--rf_params",
        type=str,
        help="Path to JSON file containing the configuration for the random forest",
        required=True
    )

    args = parser.parse_args()

    run(args)
