import numpy as np

#*******************************************************
#                   BASIC functions
#*******************************************************


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    
    for n_iter in range(max_iters):
        grad = compute_gradient_ls(y, tx, w)
        w = w - gamma*grad
        
        if n_iter % 100 == 0 or n_iter == max_iters-1:
            loss = compute_loss_mse(y, tx, w)
            #print("Current iteration={i}, training loss={l}".format(i=n_iter, l=loss))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        i = np.random.choice(np.arange(len(y)))
        # loss is the MSE
        grad = compute_gradient_ls(y[i:i+1], tx[i:i+1], w)
        w = w - gamma*grad
        
        if n_iter % 100 == 0 or n_iter == max_iters-1:
            loss = compute_loss_mse(y, tx, w)
            #print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
        
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    l1 = 2*len(tx)*lambda_
    xx = tx.T.dot(tx)
    a = (xx+ l1*np.eye(len(xx)))
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """compute the gradient of loss for Logistic Regression. Hard coded mini-batch size of 1."""
    w = initial_w
    
    # start the logistic regression
    for n_iter in range(max_iters):
        
        i = np.random.choice(np.arange(len(y)))
        
        grad = calculate_gradient_lr(y[i:i+1], tx[i:i+1], w)
        w = w - gamma*grad   
        
        # log info only at certain steps and at the last step.
        if n_iter % 1000 == 0 or n_iter == max_iters-1:
            loss = calculate_loss_lr(y, tx, w)
            #print("Current iteration={i}, training loss={l}".format(i=n_iter, l=loss))
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic Regression with SGD."""
    w = initial_w
    # start the logistic regression
    for n_iter in range(max_iters):
        
        i = np.random.choice(np.arange(len(y)))
        grad = calculate_gradient_lr_reg(y[i:i+1], tx[i:i+1], lambda_, w)
        w = w - gamma*grad   
        
        # log info only at certain steps
        if n_iter % 1000 == 0 or n_iter == max_iters-1:
            loss = calculate_loss_lr_reg(y, tx, lambda_, w)
            #print("Current iteration={i}, training loss={l}".format(i=n_iter, l=loss))
    
    return w, loss

# ***************************************************
#  Loss Functions
# ***************************************************

def compute_loss_mse(y, tx, w):
    """Calculate the mse loss."""
    return 0.5*((y-tx.dot(w))**2).mean()

def compute_loss_mse_loop(y, tx, w):
    """compute the loss mse. With for loop because of memory error"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + ((y[i]-tx[i].dot(w))**2)
    return 0.5*loss/y.shape[0]

def calculate_loss_lr(y, tx, w):
    """compute the loss: negative log likelihood. With for loop to avoid memory error of Matrix Ops."""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1+np.exp(tx[i].dot(w))) - y[i]*(tx[i].dot(w))
    return loss

def calculate_loss_lr_norm(y, tx, w):
    """compute the loss: negative log likelihood. With for loop to avoid memory error of Matrix Ops.
       This loss is normalized by # of data points to allow comparison between attempts."""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1+np.exp(tx[i].dot(w))) - y[i]*(tx[i].dot(w))
    return loss/y.shape[0]

def calculate_loss_lr_reg(y, tx, lambda_, w):
    """compute the regularized loss: negative log likelihood. With for loop because of memory error"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1 + np.exp(tx[i].dot(w))) - y[i]*(tx[i].dot(w)) + 0.5*lambda_*np.squeeze(w.T.dot(w))
    return loss

def calculate_loss_lr_reg_norm(y, tx, lambda_, w):
    """compute the regularized loss: negative log likelihood. With for loop because of memory error"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1 + np.exp(tx[i].dot(w))) - y[i]*(tx[i].dot(w)) + 0.5*lambda_*np.squeeze(w.T.dot(w))
    return loss/y.shape[0]

# ***************************************************
#  Calculate Gradient Functions
# ***************************************************

def compute_gradient_ls(y, tx, w):
    """Compute the gradient for Least Squares."""
    y = y.reshape(y.shape[0], 1)
    err = y - tx.dot(w);
    grad = -1/len(y) * tx.T.dot(err)
    
    return grad

def calculate_gradient_lr(y, tx, w):
    """compute the gradient of loss for Logistic Regression."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)

def calculate_gradient_lr_reg(y, tx, lambda_, w):
    """compute the gradient of loss for Logistic Regression."""
    return tx.T.dot(sigmoid(tx.dot(w))-y) + lambda_*w
    
# ***************************************************
#  Useful Functions
# ***************************************************
    
def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))

def standardize(x):
    ''' Standardize each column of the input. '''
    x = (x-np.mean(x, axis=0))/np.std(x, axis=0)
    return x


def build_poly (tX, degree):
    N= len(tX[0])
    new_tX = np.ones(tX.shape[0])
    for i in range(N):
        feature = tX[:,i]
        j= np.arange(1, degree+1)
        feature=feature.reshape(-1,1)
        ft = feature**(j)
        new_tX = np.column_stack((new_tX, ft))
    new_tX = np.delete(new_tX, 0, axis=1)
    return new_tX


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # randomly select ratio of inds
    n = np.floor(ratio*x.shape[0]).astype('int')
    inds = np.random.choice(x.shape[0], n, replace=False) 
    x_train = x[inds]
    y_train = y[inds]
    x_test = np.delete(x, inds, axis=0)
    y_test = np.delete(y, inds)
    
    return x_train, y_train, x_test, y_test




def calculate_accuracy (y_test, x_test, coeffs) :
    """ Calculates the accuracy of our model. """
    y_pred = x_test.dot(coeffs)
    acc = np.sum(y_pred==y_test)/len(y_test)
    print("There are {acc} % of correct predictions".format(
              acc = accuracy))

    
    
    


    
    # ***************************************************
    #  FINAL MODEL
    # ***************************************************
    
# Implementations functions
def logistic_regression_mod(y, tx, initial_w, max_iters, gamma):
    """compute the gradient of loss for Logistic Regression. Hard coded mini-batch size of 1.
       Mod version to use normalized loss function in order to compare accross runs"""
    w = initial_w
    
    # start the logistic regression
    for n_iter in range(max_iters):
        
        i = np.random.choice(np.arange(len(y)))
        
        grad = calculate_gradient_lr(y[i:i+1], tx[i:i+1], w)
        w = w - gamma*grad   
        
        # log info only at certain steps and at the last step.
        if n_iter % 1000 == 0 or n_iter == max_iters-1:
            loss = calculate_loss_lr_norm(y, tx, w)
            #print("Current iteration={i}, training loss={l}".format(i=n_iter, l=loss))
    
    return w, loss

def logistic_regression_model(y, tx, max_iters, gamma):
    """polynomial regression with different split ratios and different degrees."""
    w_init = np.zeros((tx.shape[1], 1))
    
    x_train, y_train, x_test, y_test = split_data(tx, y, ratio=0.8, seed=1)
    
    w_, loss_tr = logistic_regression_mod(y_train, x_train, w_init, max_iters, gamma)
    
    # modified helper functions returns 0 or 1
    pred = predict_labels01(w_, x_test)

    error_te = 0
    for i in range(y_test.shape[0]):
            error = np.abs(y_test[i] - pred[i])
            error_te = error_te + error    
   
    print('Proportion test error: ', error_te/y_test.shape[0])
    
    return w_

def logistic_regression_model_winit(y, tx, w_init, max_iters, gamma):
    """polynomial regression with different split ratios and different degrees.
       Here, we submit an initial weights vector obtained from previous sub-model"""
    # re-use weights from first model
    w_init = w_init
    w_init[-4:] = 0
    w_init = np.resize(w_init, (tx.shape[1],1))
    
    x_train, y_train, x_test, y_test = split_data(tx, y, ratio=0.8, seed=1)
    
    w_, loss_tr = logistic_regression_mod(y_train, x_train, w_init, max_iters, gamma)
    
    # modified helper functions returns 0 or 1
    pred = predict_labels01(w_, x_test)
    error_te = 0
    for i in range(y_test.shape[0]):
            error = np.abs(y_test[i] - pred[i])
            error_te = error_te + error    
   
    print('Proportion test error: ', error_te/y_test.shape[0])
    
    return w_

# Feature Processing
def remove_useless_cols(tx, headers):
    """Remove features deemed to have no useful information."""
    cols_remove = []
    cols_remove.append(np.where(headers == "PRI_jet_leading_phi")[0])
    cols_remove.append(np.where(headers == "PRI_jet_subleading_phi")[0])
    cols_remove.append(np.where(headers == "PRI_tau_phi")[0])
    cols_remove.append(np.where(headers == "PRI_met_phi")[0])
    cols_remove.append(np.where(headers == "PRI_lep_phi")[0])

    tx_ = np.delete(tx, cols_remove, axis = 1)
    head_ = np.delete(headers, cols_remove) 
    
    return tx_, head_

def replace_999_mean(tx, col):
    """Replace -999 values with column mean."""
    inds_missing = np.where(tx[:,col] == -999) 
    inds_good = np.delete(np.arange(tx.shape[0]), inds_missing)
    mean_ = np.mean(tx[inds_good, col])  # caluclate the mean using only non-missing values
    tx[inds_missing, col] = mean_        # replace all missing values with the mean
    
    return tx
    
    
def process_features_train(tx, headers, y, deg):
    """ Processes tx to output 3 separate datasets and targets based on
        PRI_jet_num. (For training step) """
    
    # remove features deemed useless from EDA
    tx_, head_ = remove_useless_cols(tx, headers)
    
    # replace missing values in first feature with mean
    tx_ = replace_999_mean(tx_, 0)
    
    # determine indices of each jet#
    col = np.where(head_=='PRI_jet_num')[0]
    inds0 = np.where(tx_[:, col] == 0)[0]
    inds1 = np.where(tx_[:, col] == 1)[0]
    inds2 = np.where(tx_[:, col] == 2)[0]
    inds3 = np.where(tx_[:, col] == 3)[0]


    # initialize new columns as zeros
    pri_jet0 = np.zeros(tx_.shape[0])
    pri_jet1 = np.zeros(tx_.shape[0])
    pri_jet2 = np.zeros(tx_.shape[0])
    pri_jet3 = np.zeros(tx_.shape[0])

    # set ones to appropriate columns
    pri_jet0[inds0] = 1
    pri_jet1[inds1] = 1
    pri_jet2[inds2] = 1
    pri_jet3[inds3] = 1
    
    # remove jet_num column
    tx_ = np.delete(tx_, col, axis=1)
    head_ = np.delete(head_, col)
    
    # create separate datasets
    tx_0 = tx_.copy()

    # Features to remove for reduced tx_ 
    cols_remove0 = []
    cols_remove0.append(np.where(head_ == "DER_deltaeta_jet_jet")[0][0])
    cols_remove0.append(np.where(head_ == "DER_mass_jet_jet")[0][0])
    cols_remove0.append(np.where(head_ == "DER_prodeta_jet_jet")[0][0])
    cols_remove0.append(np.where(head_ == "DER_lep_eta_centrality")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_leading_pt")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_leading_eta")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_subleading_pt")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_subleading_eta")[0][0])

    # cols to add for tx_1
    cols_add1 = []
    cols_add1.append(np.where(head_ == "PRI_jet_leading_pt")[0][0])
    cols_add1.append(np.where(head_ == "PRI_jet_leading_eta")[0][0])

    # remove features to create base dataset tx_0
    tx_0 = np.delete(tx_0, cols_remove0, axis=1)
    tx_1 = tx_0.copy()
    tx_2 = tx_0.copy()

    # add back features to end of tx_0 to create tx_1 and tx_2
    tx_1 = np.column_stack((tx_1, tx_[:, cols_add1]))
    tx_2 = np.column_stack((tx_0, tx_[:, cols_remove0]))

    # filter tx_1 and tx_2 to keep only relevant jet#
    tx_1 = tx_1[inds1, :]
    inds23 = np.append(inds2, inds3)
    tx_2 = tx_2[inds23, :]
    
    # expand features of each dataset
    tx_0 = build_poly(tx_0, deg)
    tx_1 = build_poly(tx_1, deg)
    tx_2 = build_poly(tx_2, deg)
    
    tx_0 = standardize(tx_0)
    tx_1 = standardize(tx_1)
    tx_2 = standardize(tx_2)
    
    # only add catergorical value when more than 1 jet# included
    tx_0 = np.column_stack((tx_0,pri_jet0, pri_jet1, pri_jet2, pri_jet3))
    tx_2 = np.column_stack((tx_2, pri_jet2[inds23], pri_jet3[inds23]))
    
    # addition of bias column
    tx_0 = np.c_[np.ones(tx_0.shape[0]), tx_0]
    tx_1 = np.c_[np.ones(tx_1.shape[0]), tx_1]
    tx_2 = np.c_[np.ones(tx_2.shape[0]), tx_2]
    
    # create subsets of tx_0 to predict with base model
    tx_00 = tx_0[inds0]
    tx_01 = tx_0[inds1]
    tx_02 = tx_0[inds2]
    tx_03 = tx_0[inds3]
    tx_023 = np.concatenate((tx_02, tx_03), axis=0)
    
    data = []
    data.append(tx_0)
    data.append(tx_00)
    data.append(tx_1)
    data.append(tx_01)
    data.append(tx_2)
    data.append(tx_023)
    
    # create ids for jets 2+3
    ids23 = np.append(inds2, inds3)
    
    ids = []
    ids.append(inds0)
    ids.append(inds1)
    ids.append(ids23)
    
    # create corresponding y vectors for each jet#
    y0 = y.copy()
    y1 = y.copy()
    y23 = y.copy()

    y0 = y0[inds0]
    y1 = y1[inds1]
    y23 = y23[inds23] 
    
    # Create target y values with 0,1 instead of -1,1
    y01 = y.copy()
    y01[y01 < 0] = 0
    
    y001 = y0.copy() 
    y001[y001 < 0] = 0
    
    y101 = y1 
    y101[y101 < 0] = 0

    y201 = y23
    y201[y201 < 0] = 0
    
    targets = []
    targets.append(y01)
    targets.append(y001)
    targets.append(y101)
    targets.append(y201)
    
    return  data, targets, ids

def process_features_test(tx, headers, ids, deg):
    """ Processes tx to output 3 separate datasets based on PRI_jet_num. 
        Keeps track of indices for ordering of predictions. (For test step) """
    
    # remove features deemed useless from EDA
    tx_, head_ = remove_useless_cols(tx, headers)
    
    # replace missing values in first feature with mean
    tx_ = replace_999_mean(tx_, 0)
    
    # first determine the categorical columns
    col = np.where(head_=='PRI_jet_num')[0]
    inds0 = np.where(tx_[:, col] == 0)[0]
    inds1 = np.where(tx_[:, col] == 1)[0]
    inds2 = np.where(tx_[:, col] == 2)[0]
    inds3 = np.where(tx_[:, col] == 3)[0]
    
    # initialize new columns as zeros
    pri_jet0 = np.zeros(tx_.shape[0])
    pri_jet1 = np.zeros(tx_.shape[0])
    pri_jet2 = np.zeros(tx_.shape[0])
    pri_jet3 = np.zeros(tx_.shape[0])

    # set ones to appropriate columns
    pri_jet0[inds0] = 1
    pri_jet1[inds1] = 1
    pri_jet2[inds2] = 1
    pri_jet3[inds3] = 1
    
    # remove jet_num column
    tx_ = np.delete(tx_, col, axis=1)
    head_ = np.delete(head_, col)
    
    # create separate datasets
    tx_0 = tx_.copy()

    # Features to remove for reduced tx_ 
    cols_remove0 = []
    cols_remove0.append(np.where(head_ == "DER_deltaeta_jet_jet")[0][0])
    cols_remove0.append(np.where(head_ == "DER_mass_jet_jet")[0][0])
    cols_remove0.append(np.where(head_ == "DER_prodeta_jet_jet")[0][0])
    cols_remove0.append(np.where(head_ == "DER_lep_eta_centrality")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_leading_pt")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_leading_eta")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_subleading_pt")[0][0])
    cols_remove0.append(np.where(head_ == "PRI_jet_subleading_eta")[0][0])

    # cols to add for tx_1
    cols_add1 = []
    cols_add1.append(np.where(head_ == "PRI_jet_leading_pt")[0][0])
    cols_add1.append(np.where(head_ == "PRI_jet_leading_eta")[0][0])

    # remove features from tx_ to create base dataset tx_0
    tx_0 = np.delete(tx_0, cols_remove0, axis=1)
    tx_1 = tx_0.copy()
    tx_2 = tx_0.copy()

    # add back features to end of tx_0 to create tx_1 and tx_2
    tx_1 = np.column_stack((tx_1, tx_[:, cols_add1]))
    tx_2 = np.column_stack((tx_0, tx_[:, cols_remove0]))

    # filter tx_1 and tx_2 to keep only relevant jet#s
    tx_1 = tx_1[inds1, :]
    inds23 = np.append(inds2, inds3)
    tx_2 = tx_2[inds23, :]

    # expand features and standardize
    tx_0 = build_poly(tx_0, deg)
    tx_1 = build_poly(tx_1, deg)
    tx_2 = build_poly(tx_2, deg)

    tx_0 = standardize(tx_0)
    tx_1 = standardize(tx_1)
    tx_2 = standardize(tx_2)
    
    # only add catergorical value when more than 1 jet# included
    tx_0 = np.column_stack((tx_0,pri_jet0, pri_jet1, pri_jet2, pri_jet3))
    tx_2 = np.column_stack((tx_2, pri_jet2[inds23], pri_jet3[inds23]))

    # add bias feature
    tx_0 = np.c_[np.ones(tx_0.shape[0]), tx_0]
    tx_1 = np.c_[np.ones(tx_1.shape[0]), tx_1]
    tx_2 = np.c_[np.ones(tx_2.shape[0]), tx_2]

    
    # separate ids to restore order after separately predicting 
    ids0 = ids[inds0]
    ids1 = ids[inds1]
    ids2 = ids[inds2]
    ids3 = ids[inds3]
    ids23 = np.append(ids2, ids3)
    
    # create subsets of tx_0 to predict with base model
    tx_00 = tx_0[inds0]
    tx_01 = tx_0[inds1]
    tx_02 = tx_0[inds2]
    tx_03 = tx_0[inds3]
    tx_023 = np.concatenate((tx_02, tx_03), axis=0)
    
    
    data = []
    data.append(tx_0)
    data.append(tx_00)
    data.append(tx_1)
    data.append(tx_01)
    data.append(tx_2)
    data.append(tx_023)
    
    ids = []
    ids.append(ids0)
    ids.append(ids1)
    ids.append(ids23)
    
    return  data, ids

# Create predictions

def predict_labels01_comb(weights1, data1, weights2, data2):
    """Generates class predictions (0,1) given weights, and a test data matrix.
       Uses a combination of two sub-model predictions."""
    y_pred1 = np.dot(data1, weights1)
    y_pred2 = np.dot(data2, weights2)
    y_pred = 0.6*y_pred1 + 0.4*y_pred2
    y_pred[np.where(y_pred <= 0)] = 0
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def predict_labels01(weights, data):
    """Generates class predictions (0,1) given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = 0
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_predictions(weights, data, ids):
    '''Takes list of weights and list of data matrices to create final predictions.'''
    weights0 = weights[0]
    weights1 = weights[1]
    weights2 = weights[2]

    tx_0 = data[0]
    tx_00 = data[1]
    tx_1 = data[2]
    tx_01 = data[3]
    tx_2 = data[4]
    tx_02 = data[5]
    
    ids0 = ids[0]
    ids1 = ids[1]
    ids23 = ids[2]

    # predict labels for jet=0
    y_pred0 = predict_labels01(weights0, tx_00)
    y_pred0[y_pred0 == 0] = -1

    # predictions for jet=1
    y_pred1 = predict_labels01_comb(weights1, tx_1, weights0, tx_01)
    y_pred1[y_pred1 == 0] = -1

    # predictions for jet=23
    y_pred2 = predict_labels01_comb(weights2, tx_2, weights0, tx_02)
    y_pred2[y_pred2 == 0] = -1
    
    y_pred = np.vstack((y_pred0, y_pred1, y_pred2))

    ids_final = np.append(ids0, ids1)
    ids_final = np.append(ids_final, ids23)
    ids_final = ids_final[:,np.newaxis]

    # concatenate ids with preds
    y_pred_ids = np.concatenate((ids_final, y_pred), axis=1)

    # reorder predictions
    y_pred_order = y_pred_ids[y_pred_ids[:,0].argsort()]

    # remove ids column now that order is restored
    y_pred_final = y_pred_order[:,1]
    y_pred_final = y_pred_final[:, np.newaxis]
    
    return y_pred_final

# To calculate log-likelihood loss with final model

def create_predictions_loss(weights, data, ids):
    '''Takes list of weights and list of data matrices to create predictions.
       Here we output the loss without the thresholding to 0 or 1 in order
       to use it in the negative log-likelihood loss funciton.'''
    weights0 = weights[0]
    weights1 = weights[1]
    weights2 = weights[2]

    tx_0 = data[0]
    tx_00 = data[1]
    tx_1 = data[2]
    tx_01 = data[3]
    tx_2 = data[4]
    tx_02 = data[5]
    
    ids0 = ids[0]
    ids1 = ids[1]
    ids23 = ids[2]

    # predict labels for jet=0
    y_pred0 = predict_labels01_loss(weights0, tx_00)

    # predictions for jet=1
    y_pred1 = predict_labels01_comb_loss(weights1, tx_1, weights0, tx_01)

    # predictions for jet=23
    y_pred2 = predict_labels01_comb_loss(weights2, tx_2, weights0, tx_02)
    
    y_pred = np.vstack((y_pred0, y_pred1, y_pred2))

    ids_final = np.append(ids0, ids1)
    ids_final = np.append(ids_final, ids23)
    ids_final = ids_final[:,np.newaxis]

    # concatenate ids with preds
    y_pred_ids = np.concatenate((ids_final, y_pred), axis=1)

    # reorder predictions
    y_pred_order = y_pred_ids[y_pred_ids[:,0].argsort()]

    # remove ids column now that order is restored
    y_pred_final = y_pred_order[:,1]
    y_pred_final = y_pred_final[:, np.newaxis]
    
    return y_pred_final

def predict_labels01_comb_loss(weights1, data1, weights2, data2):
    """Generates predictions given weights, and data matrices.
       Here we do not threshold to 0,1"""
    y_pred1 = np.dot(data1, weights1)
    y_pred2 = np.dot(data2, weights2)
    y_pred = 0.6*y_pred1 + 0.4*y_pred2
    
    return y_pred

def predict_labels01_loss(weights, data):
    """Generates predictions given weights, and data matrices.
       Here we do not threshold to 0,1"""
    y_pred = np.dot(data, weights)
    
    return y_pred

def calculate_loss_lr_model(y, pred):
    """compute the loss: negative log likelihood. With for loop to 
       avoid memory error of Matrix Ops.  Modified to work with our 
       final model. y must be between 0 and 1"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1+np.exp(pred[i])) - y[i]*(pred[i])
    return loss/y.shape[0]

# ********************************
# Cross Validation
# ********************************

def cv_model(tX, headers, y, degree, tX_test, y_test, headers_test):
    # process features for to train the model
    data, targets, ids = process_features_train(tX, headers, y, degree)
    
    # train base model 
    w_1 = logistic_regression_demo(targets[0], data[0], max_iters=10000, gamma=0.01)

    # train jet=1 model using base model weights as initial weights
    w_2 = logistic_regression_demo_winit(targets[2], data[2], w_1, max_iters=10000, gamma=0.01)

    # train jet=2/3 model using base model weights as initial weights
    w_3 = logistic_regression_demo_winit(targets[3], data[4], w_1, max_iters=10000, gamma=0.01)
    
    

    # process test set
    data, targets, ids = process_features_train(tX_test, headers, y_test, degree)

    # create Predictions
    weights = [w_1, w_2, w_3]
    #y_pred_final = create_predictions(weights, data, ids)
    
    # calcualte the loss, but first calculate the predictions for the loss (0 1 type)
    pred_loss = create_predictions_loss(weights, data, ids)
    
    loss = calculate_loss_lr_model(targets[0], pred_loss)
    
    # map the predictions into -1 and 1
    y_pred_final = pred_loss.copy()
    y_pred_final[np.where(y_pred_final <= 0.5)] = -1
    y_pred_final[np.where(y_pred_final > 0.5)] = 1
    
    acc = np.sum(y_pred_final.T==y_test)/len(y_test)
    
    
    return loss, acc

def cross_validation(y, x, ids, degrees):
    seed = 1
    k_fold = 4
    
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss = np.zeros((degrees, k_fold))
    accuracy = np.zeros((degrees, k_fold))
    
    
    for degree in np.arange(degrees)+1:
        loss_d = []
        acc_d = []
        for k in range(k_fold):
            #****************************************************
            # Form train and test sets
            x_tr = x[k_indices[np.arange(len(k_indices))!=k].ravel()]
            x_te = x[k_indices[k]]

            y_tr = y[k_indices[np.arange(len(k_indices))!=k].ravel()]
            y_te = y[k_indices[k]]

            ids_tr = ids[k_indices[np.arange(len(k_indices))!=k].ravel()]
            ids_te = ids[k_indices[k]]

            #****************************************************
            # form data with polynomial degree
            print("Fold number: {f}, polynomial degree: {d}".format(f=k+1, d=degree))

            #****************************************************

            lo, acc = cv_model(x_tr, headers, y_tr, degree, x_te, y_te, headers)
            
            loss_d.append(lo[0])
            acc_d.append(acc)
        loss[degree-1] = np.array(loss_d)
        accuracy[degree-1] = np.array(acc_d)
    
    return loss, accuracy

# ********************************
# Bias - Variance Plots
# ********************************

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        #label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        #label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    #plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.legend(loc=2)
    plt.title("Bias-Variance Decomposition - Baseline Model")
    plt.savefig("bias_variance_base_model")
    
def bias_variance_baseline():
    """The entry."""
    # define parameters
    seeds = range(20)
    ratio_train = 0.8
    degrees = range(1, 10)
    gamma = 0.01
    max_iters = 10000
    
    # define list to store the variable
    loss_tr = np.empty((len(seeds), len(degrees)))
    loss_te = np.empty((len(seeds), len(degrees)))
    
    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
     
        x_train, y_train, x_test, y_test = split_data(tX, y, ratio_train, seed)

  
        for index_degree, degree in enumerate(degrees):
            data_tr, targets_tr, ids_tr = process_features_train(x_train, headers, y_train, degree)
            data_te, targets_te, ids_te = process_features_train(x_test, headers, y_test, degree)
           
            w_, losstr= logistic_regression_model(targets_tr[0], data_tr[0], max_iters, gamma)
            losste = calculate_loss_lr_norm(targets_te[0], data_te[0], w_)
            
            loss_tr[index_seed, index_degree] = losstr
            loss_te[index_seed, index_degree] = losste

    bias_variance_decomposition_visualization(degrees, loss_tr, loss_te)
    
    return degrees, loss_tr, loss_te

def bias_variance_model():
    """The entry."""
    # define parameters
    seeds = range(20)
    ratio_train = 0.8
    degrees = range(1, 10)
    
    # define list to store the variable
    loss_tr_model = np.empty((len(seeds), len(degrees)))
    loss_te_model = np.empty((len(seeds), len(degrees)))
    
    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
 
        x_train, y_train, x_test, y_test = split_data(tX, y, ratio_train, seed)

        for index_degree, degree in enumerate(degrees):
            data_tr, targets_tr, ids_tr = process_features_train(x_train, headers, y_train, degree)
            data_te, targets_te, ids_te = process_features_train(x_test, headers, y_test, degree)
            
            w_1, _ = logistic_regression_model(targets_tr[0], data_tr[0], max_iters=15000, gamma=0.01)
            w_2 = logistic_regression_demo_winit(targets_tr[2], data_tr[2], max_iters=10000, gamma=0.01)
            w_3 = logistic_regression_demo_winit(targets_tr[3], data_tr[4], max_iters=10000, gamma=0.01)
            weights = [w_1, w_2, w_3]
            
            pred_loss_tr = create_predictions_loss(weights, data_tr, ids_tr)
            loss_tr = calculate_loss_lr_model(targets_tr[0], pred_loss_tr) 
            
            pred_loss_te = create_predictions_loss(weights, data_te, ids_te)
            loss_te = calculate_loss_lr_model(targets_te[0], pred_loss_te)
            
            loss_tr_model[index_seed, index_degree] = loss_tr
            loss_te_model[index_seed, index_degree] = loss_te

    bias_variance_decomposition_visualization(degrees, loss_tr_model, loss_te_model)
    
    return degrees, loss_tr_model, loss_te_model