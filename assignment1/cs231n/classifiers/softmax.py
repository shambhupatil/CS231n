from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(X.shape[0]):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores)
        prob = exp_scores[y[i]] / np.sum(exp_scores)
        loss += -np.log(prob)
        
        for j in range(num_class):
            p = exp_scores[j]/np.sum(exp_scores)
            dW[:,j] += X[i].T.dot((p - (j == y[i])))
    loss /= X.shape[0]
    loss += reg * np.sum(W*W)
    dW /= X.shape[0]
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    exp_scores = np.exp(scores)
    loss_vector = exp_scores[range(X.shape[0]),y] / np.sum(exp_scores,axis = 1)
    loss_vector = np.log(loss_vector)
    loss = -np.sum(loss_vector)
    p = exp_scores / np.sum(exp_scores,axis = 1).reshape([exp_scores.shape[0],1])
    p[range(X.shape[0]),y] -= 1
    dW = X.T.dot(p)
    loss /= X.shape[0]
    loss += reg * np.sum(W*W)
    dW /= X.shape[0]
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
