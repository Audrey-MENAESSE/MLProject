import numpy as np

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

def PCA_bias (tX) :
    m, n = tX.shape
    dataset=tX
    dataset -= tX.mean(axis=0)
    R = np.cov(dataset, rowvar=False)
    evals, evects = np.linalg.eigh(R)
    # Sorting the eigenvalues in a decreasing order
    index = np.argsort(evals)[::-1]
    evals = evals[index]
    evects = evects[:,index]
    #evects = evects[:, :n_features]
    data = np.dot(evects.T, dataset.T).T
    
    # Adding the bias term of the model
    data_ones = np.c_[np.ones((len(tX), 1)), data]
    
    return (data_ones)

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

def arrange_data (tX, headers, degree):
    data = replace_missing (tX)
    data, cat = cat_features (data, headers)    
    data = build_poly(data, degree)
    data = standardize (data)
    data = np.c_[data, cat]
    # data = PCA_bias(data)
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
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    # randomly select ratio of inds
    n = np.floor(ratio*x.shape[0]).astype('int')
    inds = np.random.choice(x.shape[0], n, replace=False) 
    x_train = x[inds]
    y_train = y[inds]
    x_test = np.delete(x, inds, axis=0)
    y_test = np.delete(y, inds)
    
    return x_train, y_train, x_test, y_test


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            

def standardize(x):
    ''' Standardize each column of the input. '''
    x = (x-np.mean(x, axis=0))/np.std(x, axis=0)
    return x


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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for iter in range(max_iters):
        
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            minibatch_y = minibatch_y[:, np.newaxis]
            grad = calculate_gradient_lr(minibatch_y, minibatch_tx, w)
            w = w - gamma*grad   
        
        # log info only at certain steps and at the last step.
        if iter % 100 == 0 or iter == max_iters-1:
            loss = calculate_loss_lr(y, tx, w)
            print("Current iteration={i}, training loss={l}".format(i=iter, l=loss))
    
    return w, loss


    # ***************************************************
    #  Regularized Logistic Regression Functions
    # ***************************************************
    

def calculate_loss_lr_reg(y, tx, lambda_, w):
    """compute the regularized loss: negative log likelihood. With for loop because of memory error"""
    loss = 0
    for i in range(y.shape[0]):
        loss = loss + np.log(1+np.exp(tx[i].dot(w))) - y[i]*(tx[i].dot(w)) + 0.5*lambda_*np.squeeze(w.T.dot(w))
    return loss


def calculate_gradient_lr_reg(y, tx, lambda_, w):
    """compute the gradient of loss for Logistic Regression."""
    return tx.T.dot(sigmoid(tx.dot(w))-y) + lambda_*w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic Regression with SGD."""
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            minibatch_y = minibatch_y[:, np.newaxis]
            grad = calculate_gradient_lr_reg(minibatch_y, minibatch_tx, lambda_, w)
            w = w - gamma*grad   
        
        # log info only at certain steps
        if iter % 100 == 0 or iter == max_iters-1:
            loss = calculate_loss_lr_reg(y, tx, lambda_, w)
            print("Current iteration={i}, training loss={l}".format(i=iter, l=loss))
    
    return w, loss

def calculate_accuracy (y_test, x_test, coeffs) :
    y_pred = x_test.dot(coeffs)
    N = len (y_test)
    T = 0
    F = 0
    for i in range (N) :
        if y_test[i]*y_pred[i]>0:
            T+=1
        else : 
            F+=1
    accuracy = 100*T/N
    print("There are {acc} % of correct predictions".format(
              acc = accuracy))
    
calculate_accuracy(y_test, x_test, coeffs)