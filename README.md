# ClimateAi_challenge

- Coding challenge of predicting river flow data 7 days ahead
- By Diego Alfaro, 03-2022

## General description of package
This repository contains python code and datasets related to the coding challenge. Jupyter notebooks are included to aide visualizations, and are annotated to show analyses and basic conclusions. Python scripts and yaml files allow data processing and Random Forest model training, with module integration, reproducibility, reusablility and environment management enabled through MLflow/conda/hydra configurations.

## MLflow components
The main.py script is meant to be run through MLflow (see "Running the training package" below), which helps to maintain environments from each component separate from the code (environment control, reusability and reproducibility). The components that are run from main.py are:

- process_data: processes the NetCDF data and outputs a pd.DataFrame with features to be used for test/training. The output is saved in the ./Data folder
- train_model: reads the train-test dataset created by process_data, training and saving a joblib Random Forest model. The model is saved in the ./Models folder

## Running the model training package
On the main.py directory, use the following command to run the package with default configurations:
```console
mlflow run .
```

The default configuration is found in the config.yaml file, and it is managed through hydra. The parameters can be overwritten by specifying them in the command line as follow (for n_estimators only):
```console
mlflow run -P n_estimators=500
```

Note that MLflow creates the necessary virtual enviroments for each component, which are defined in each component's conda.yml file. 

## Dependencies
MLflow and conda are required. Other packages are managed automatically through virtual environment creation. 
