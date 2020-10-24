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
            loss = compute_loss_mse2(y, tx, w)
            print("Current iteration={i}, training loss={l}".format(i=n_iter, l=loss))

    return loss, w


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        (y, tx) = np.random.choice(np.arange(len(y)))
        # loss is the MSE
        grad = compute_gradient_ls(y, tx, w)
        w = w - gamma*grad
        
        if n_iter % 100 == 0 or n_iter == max_iters-1:
            loss = compute_loss_mse(y, tx, w)
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
        
    return loss, w


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
    return w


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
            print("Current iteration={i}, training loss={l}".format(i=n_iter, l=loss))
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic Regression with SGD."""
    w = initial_w
    # start the logistic regression
    for n_iter in range(max_iters):
        
        i = np.random.permutation(np.arange(0, tx.shape[0]))[0]
        grad = calculate_gradient_lr_reg(y[i:i+1], tx[i:i+1], lambda_, w)
        w = w - gamma*grad   
        
        # log info only at certain steps
        if n_iter % 1000 == 0 or n_iter == max_iters-1:
            loss = calculate_loss_lr_reg(y, tx, lambda_, w)
            print("Current iteration={i}, training loss={l}".format(i=n_iter, l=loss))
    
    return w, loss

    
# ***************************************************
#  Useful Functions
# ***************************************************
""" Feature Processing functions """
# Replacing the missing values by the average 
def replace_missing (tX):
    for i in range(len(tX[0])):
        average = np.mean(tX[tX[:,i]!=-999, i])
        inds = np.where(tX[:,i]==-999)
        tX[inds, i] = average
    return tX

def cat_features (tX, headers) :
    # Finding the number of the column containing the categorical feature
    col = np.where(headers=='PRI_jet_num')[0]
    
    # The possible entries of feature PRI_jet_num are 0, 1, 2, 3
    inds0 = np.where(tX[:, col] == 0)[0]
    inds1 = np.where(tX[:, col] == 1)[0]
    inds2 = np.where(tX[:, col] == 2)[0]
    inds3 = np.where(tX[:, col] == 3)[0]

    # initialize new columns as zeros
    pri_jet0 = np.zeros(tX.shape[0])
    pri_jet1 = np.zeros(tX.shape[0])
    pri_jet2 = np.zeros(tX.shape[0])
    pri_jet3 = np.zeros(tX.shape[0])

    # set ones to appropriate columns
    pri_jet0[inds0] = 1
    pri_jet1[inds1] = 1
    pri_jet2[inds2] = 1
    pri_jet3[inds3] = 1

    # create a vector with the new categorical features
    new_cat = np.column_stack((pri_jet0, pri_jet1, pri_jet2, pri_jet3))
    
    # delete original jet_num column
    tX = np.delete(tX, col, axis=1)
    headers = np.delete(headers, col)
    
    # append new columns
    #tX = np.column_stack((tX,pri_jet0, pri_jet1, pri_jet2, pri_jet3))
    new_heads = ['pri_jet_num0', 'pri_jet_num1', 'pri_jet_num2', 'pri_jet_num3']
    #head_ = np.append(head_, new_heads)
    
    return tX, new_cat

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
    return new_tX



def arrange_data (tX, headers, degree):
    data = replace_missing (tX)
    data, cat = cat_features (data, headers)    
    data = build_poly(data, degree)
    data = standardize (data)
    data = np.c_[data, cat]
    return data



""" Data splitting functions """

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


    # ***************************************************
    #  Least squares with Gradient Descent
    # ***************************************************

def compute_gradient_ls(y, tx, w):
    """Compute the gradient."""

    err = y[:, np.newaxis] - tx.dot(w);
    grad = -1/len(y) * tx.T.dot(err)
    
    return grad


def compute_loss_mse(y, tx, w):
    """Calculate the mse loss."""
    return 0.5*((y-tx.dot(w))**2).mean()

def compute_loss_mse_loop(y, tx, w):
    """compute the loss mse. With for loop because of memory error"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + ((y[i]-tx[i].dot(w))**2)
    return 0.5*loss/y.shape[0]




def calculate_loss_mse2(y, tx, w):
    """compute mse loss: for loop because of memory error"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + 0.5*((y[i]-tx[i].dot(w))**2)
    return loss/y.shape[0]


    # ***************************************************
    #  Logistic Regression Functions
    # ***************************************************

    
def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))


def calculate_loss_lr(y, tx, w):
    """compute the loss: negative log likelihood. With for loop to avoid memory error of Matrix Ops."""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1+np.exp(tx[i].dot(w))) - y[i]*(tx[i].dot(w))
    return loss


def calculate_gradient_lr(y, tx, w):
    """compute the gradient of loss for Logistic Regression."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)




    # ***************************************************
    #  Regularized Logistic Regression Functions
    # ***************************************************
    

def calculate_loss_lr_reg(y, tx, lambda_, w):
    """compute the regularized loss: negative log likelihood. With for loop because of memory error"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1+np.exp(tx[i].dot(w))) - y[i]*(tx[i].dot(w)) + 0.5*lambda_*np.squeeze(w.T.dot(w))
    loss = loss/y.shape[0]
    return loss


def calculate_gradient_lr_reg(y, tx, lambda_, w):
    """compute the gradient of loss for Logistic Regression."""
    return tx.T.dot(sigmoid(tx.dot(w))-y) + lambda_*w



def calculate_accuracy (y_test, x_test, coeffs) :
    y_pred = x_test.dot(coeffs)
    acc = np.sum(y_pred==y_test)/len(y_test)
    print("There are {acc} % of correct predictions".format(
              acc = accuracy))

    
    # ***************************************************
    #  MEGA MODEL
    # ***************************************************
    

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

def logistic_regression_demo(y, tx, max_iters, gamma):
    """polynomial regression with different split ratios and different degrees."""
    w_init = np.zeros((tx.shape[1], 1))
    
    x_train, y_train, x_test, y_test = split_data(tx, y, ratio=0.8, seed=1)
    
    w_, loss_tr = logistic_regression(y_train, x_train, w_init, max_iters, gamma)
    
    # modified helper functions returns 0 or 1
    pred = predict_labels01(w_, x_test)

    error_te = 0
    for i in range(y_test.shape[0]):
            error = np.abs(y_test[i] - pred[i])
            error_te = error_te + error    
   
    print('Proportion test error: ', error_te/y_test.shape[0])
    
    return w_

def logistic_regression_demo_winit(y, tx, max_iters, gamma):
    """polynomial regression with different split ratios and different degrees."""
    w_init = w_1.copy()
    # re-use weights from first model
    w_init[-4:] = 0
    w_init.resize(tx.shape[1],1)
    
    x_train, y_train, x_test, y_test = split_data(tx, y, ratio=0.8, seed=1)
    
    w_, loss_tr = logistic_regression(y_train, x_train, w_init, max_iters, gamma)
    
    # modified helper functions returns 0 or 1
    pred = predict_labels01(w_, x_test)
    
    error_te = 0
    for i in range(y_test.shape[0]):
            error = np.abs(y_test[i] - pred[i])
            error_te = error_te + error    
   
    print('Proportion test error: ', error_te/y_test.shape[0])
    
    return w_

def process_features_test(tx, headers, ids, deg):
    
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

def create_predictions(weights, data, ids):
    '''Takes list of weights and list of data matrices to create predictions.'''
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
    y_pred0[y_pred0 < 0.6] = -1

    # predictions for jet=1
    y_pred1 = predict_labels01_comb(weights1, tx_1, weights0, tx_01)
    y_pred1[y_pred1 < 0.6] = -1

    # predictions for jet=23
    y_pred2 = predict_labels01_comb(weights2, tx_2, weights0, tx_02)
    y_pred2[y_pred2 < 0.6] = -1
    
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

def predict_labels01_comb(weights1, data1, weights2, data2):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred1 = np.dot(data1, weights1)
    y_pred2 = np.dot(data2, weights2)
    y_pred = 0.6*y_pred1 + 0.4*y_pred2
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def predict_labels01(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred