import numpy as np

from implementations import *
from proj1_helpers import *

# Load data
DATA_TRAIN_PATH = 'train.csv' 
y, tX, ids, headers = load_csv_data(DATA_TRAIN_PATH) # Modified the load_csv_data to also give headers


# process features for to train the model
degree = 2  # hardcoded value found from cross-validation
data, targets, ids = process_features_train(tX, headers, y, degree)


# train base model 
w_1 = logistic_regression_model(targets[0], data[0], max_iters=200000, gamma=0.01)

# train jet=1 model using base model weights as initial weights
w_2 = logistic_regression_model_winit(targets[2], data[2], w_1, max_iters=30000, gamma=0.01)

# train jet=2/3 model using base model weights as initial weights
w_3 = logistic_regression_model_winit(targets[3], data[4], w_1, max_iters=30000, gamma=0.01)


# Load test prediction data
DATA_TEST_PATH = 'test.csv' 
_, tX_test, ids_test, headers = load_csv_data(DATA_TEST_PATH)

# process test set
data, ids = process_features_test(tX_test, headers, ids_test, degree)

# create Predictions
weights = [w_1, w_2, w_3]
y_pred_final = create_predictions(weights, data, ids)

# create output csv file
OUTPUT_PATH = 'predictions.csv' # TODO: fill in desired name of output file for submission
create_csv_submission(ids_test, y_pred_final, OUTPUT_PATH)
