<<<<<<< HEAD
# Machine learning Project 1 : Prediction of Higgs Boson creation during LHC collisions

##### Course
Machine Learning CS-433, École Polytechnique Fédérale de Lausanne 

##### Authors
* Acquati Francesco
* Blackburn Michaël
* Ménaësse Audrey

This repository contains the relevant code to train a model ... Higgs Boson Challenge

## Installation
A Python environment is required to run the scripts. More information can be found [here] (https://docs.python.org/fr/3/using/index.html).

## Usage
To create the csv file containing the results of the predictions, run the `run.py` script. 
It requires Python NumPy package as well as functions from `implementations.py` and `proj1_helpers.py`.

The `implementations.py` file also contains the implementations of standard machine learning methods as well as cross-validation and bias-variance analysis.

## Code architecture

The `zip` file contains:
* A README
* `proj1_helpers.py`
* `implementations.py`
* `run.py`

### proj1_helpers
This file contains the functions to read the train and test data from `csv` files, predict the labels from weights and a dataset and write a `csv` file with the predictions.

### implementations
This file contains useful functions to run the `run.py` script and is organized as follows:
* Basic functions: `least_squares_GD`, `least_squares_SGD`, `least_squares`, `ridge_regression`, `logistic_regression` and `reg_logistic_regression`.
* Loss functions: `compute_loss_mse`, `compute_loss_mse_loop`, `calculate_loss_lr`, `calculate_loss_lr_norm`, `calculate_loss_lr_reg` and `calculate_loss_lr_reg_norm`.
* Calculate Gradient Functions: `compute_gradient_ls`, `calculate_gradient_lr` and `calculate_gradient_lr_reg`.
* Useful Functions: `sigmoid`, `standardize`, `build_poly`, `split_data` and `calculate_accuracy`.
* Final Model: `logistic_regression_mod`, `logistic_regression_demo`, `logistic_regression_demo_winit`, `remove_useless_cols`, `replace_999_mean`, `process_features_train`, `process_features_test`, `predict_labels01_comb`, `predict_labels01`, `create_predictions`, `create_predictions_loss`, `predict_labels01_comb_loss`, `predict_labels01_loss` and `calculate_loss_lr_model`. 
* Cross-Validation: `cv_model` and `cross_validation`. 
=======
# Machine learning Project 1 : Prediction of Higgs Boson creation during LHC collisions

##### Course
Machine Learning CS-433, École Polytechnique Fédérale de Lausanne 

##### Authors
* Acquati Francesco
* Blackburn Michaël
* Ménaësse Audrey

This repository contains the relevant code to train a model ... Higgs Boson Challenge

## Installation
A Python environment is required to run the scripts. More information can be found [here] (https://docs.python.org/fr/3/using/index.html).

## Usage
To create the csv file containing the results of the predictions, run the `run.py` script. 
It requires Python NumPy package as well as functions from `implementations.py` and `proj1_helpers.py`.

The `implementations.py` file also contains the implementations of standard machine learning methods as well as cross-validation and bias-variance analysis.

## Code architecture

The `zip` file contains :
* A README
* `proj1_helpers.py`
* `implementations.py`
* `run.py`

### `proj1_helpers.py`
>>>>>>> 7224a8390d68297c4e569006c7a5014eb90da9b6
