import numpy as np

#### Useful functions #####

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


#### Logistic Regression Functions ####

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