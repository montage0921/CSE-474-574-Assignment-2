import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1+np.exp(-z)) # np.exp works for scalar, vector and matrix


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    # initialization
    train_validation_data=[]
    train_validation_labels=[]

    # For Training and Validation
    for digit in range(10):
        # mat contain numpy array named as train0, train1,...train9 & test0,test1,...test9
        key=f"train{digit}"
        data=mat[key]
        # labels is a vertical column with same number of rows of data, and fill it with digit
        labels=np.full((data.shape[0],1),digit) 
        
        train_validation_data.append(data)
        train_validation_labels.append(labels)
    
    # merge all data and labels into one single set
    train_validation_data=np.concatenate(train_validation_data)
    train_validation_labels=np.concatenate(train_validation_labels)

    # shuffle them
    indices=np.random.permutation(train_validation_data.shape[0])
    train_validation_data=train_validation_data[indices]
    train_validation_labels=train_validation_labels[indices]

    # split training set and validation set
    # for data
    train_data=train_validation_data[:50000]
    validation_data=train_validation_data[50000:]
    # for label
    train_label=train_validation_labels[:50000]
    validation_label=train_validation_labels[50000:]

    # For Testing Data
    test_data=[]
    test_label=[]
    for digit in range(10):
        key=f"test{digit}"
        data=mat[key]
        label=np.full((data.shape[0],1),digit)
        test_data.append(data)
        test_label.append(label)
    
    test_data=np.concatenate(test_data)
    test_label=np.concatenate(test_label)

    # flatten labels
    train_label=train_label.flatten()
    validation_label=validation_label.flatten()
    test_label=test_label.flatten()

    # Feature selection
    # Your code here.
    diff_rows=np.diff(train_data,axis=0)
    non_constant_col=np.any(diff_rows,axis=0) # boolean array to indicate if a column contain all same values

    # select only non-constant columns
    train_data=train_data[:,non_constant_col]
    validation_data=validation_data[:,non_constant_col]
    test_data=test_data[:,non_constant_col]

    # save indices of feature that we used
    selected_feature_indices = np.where(non_constant_col)[0]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    N=training_data.shape[0] # number of training data
 
    training_data=np.hstack((training_data,np.ones((N,1)))) # add bias

    intput_to_hidden=np.dot(training_data,w1.T)
    output_from_hidden=sigmoid(intput_to_hidden)

    output_from_hidden=np.hstack((output_from_hidden,np.ones((output_from_hidden.shape[0],1)))) # add bias

    input_to_class=np.dot(output_from_hidden,w2.T)
    output=sigmoid(input_to_class)

    #one hot encoding
    y=np.zeros((N,n_class))
    y[np.arange(N), training_label.astype(int)] = 1

    # NLL loss function
    obj_val=-np.sum(y*np.log(output)+(1-y)*np.log(1-output))/N

    # with regularization
    reg=(lambdaval/(2*N))*(np.sum(w1**2)+np.sum(w2**2))
    obj_val+=reg

    # calculate residual for output
    residual_output=output-y

    # calculate gradient for hidden to output layer
    grad_w2=(np.dot(residual_output.T,output_from_hidden)+lambdaval*w2)/N

    # calculate residual for hidden layer
    residual_hidden = (1 - output_from_hidden[:, :-1]) * output_from_hidden[:, :-1] * np.dot(residual_output, w2[:, :-1])

  
    # calculate gradient for training to hidden layer
    grad_w1=(np.dot(residual_hidden.T,training_data)+lambdaval*w1)/N



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([]) 

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    return labels


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


