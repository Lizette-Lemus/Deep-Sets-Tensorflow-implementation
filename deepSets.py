import numpy as np
import tensorflow as tf
from dataEntropy import *
import matplotlib.pyplot as plt

tf.set_random_seed(0)

def init_weights(shape):
    """ Initializes the weights of one layer using heuristic
    Arguments:
    shape - shape of layer
    Returns:
    weights - tf variable with weights
    """
    weights = tf.random_uniform(shape, minval= -1.0/np.sqrt(
            shape[0]), maxval = 1.0/np.sqrt(shape[0]))
    return tf.Variable(weights)

def init_all_weights(dim):
    """Initializes all the weights variables
    Arguments:
    dim - dimension of the data
    """
    h_size = 250
    rho_size = 75
    W1 = init_weights((dim, h_size))
    b1 = tf.get_variable("b1", [1, h_size], initializer = tf.zeros_initializer())
    W2 = init_weights((h_size, h_size))
    b2 = tf.get_variable("b2", [1, h_size], initializer = tf.zeros_initializer())
    W3 = init_weights((h_size, h_size))
    b3 = tf.get_variable("b3", [1, h_size], initializer = tf.zeros_initializer())
    W4 = init_weights((h_size, rho_size))
    b4 = tf.get_variable("b4", [1, rho_size], initializer = tf.zeros_initializer())
    W5 = init_weights((rho_size, rho_size))
    b5 = tf.get_variable("b5", [1, rho_size], initializer = tf.zeros_initializer())
    W6 = init_weights((rho_size, 1))
    b6 = tf.get_variable("b6", [1, 1], initializer = tf.zeros_initializer())
    weights = {"W1": W1,
               "b1": b1,
               "W2": W2,
               "b2": b2,
               "W3": W3,
               "b3": b3,
               "W4": W4,
               "b4": b4,
               "W5": W5,
               "b5": b5,
               "W6": W6,
               "b6": b6}
    return weights

def forwardprop(X, weights, segment_ids):
    """Implements the forward propagation for the model: LINEAR-> RELU 
    LINEAR->RELU->LINEAR->MEAN EMBEDDING -> LINEAR ->RELU -> LINEAR ->RELU
    ->LINEAR ->RELU
    Arguments:
    X -- a placeholder for the sets stacked into a matrix
    weights -- python dictionary containing weights
    segment_ids -- a placeholder to identify the elements with their set
    Returns:
    Z6 -- the output of the last linear unit
    """
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
    W3 = weights["W3"]
    b3 = weights["b3"]
    W4 = weights["W4"]
    b4 = weights["b4"]
    W5 = weights["W5"]
    b5 = weights["b5"]
    W6 = weights["W6"]
    b6 = weights["b6"]

    Z1 = tf.add(tf.matmul(X, W1), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(A1, W2), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(A2, W3), b3)
    mean_emb = tf.segment_mean(Z3, segment_ids)
    Z4 = tf.add(tf.matmul(mean_emb, W4), b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(A4, W5), b5)
    A5 = tf.nn.relu(Z5) 
    Z6 = tf.add(tf.matmul(A5, W6), b6)    
    return tf.reshape(Z6, [-1])

def create_placeholders(dim):
    """Creates placeholders for the tensorflow session
    Arguments:
    dim -- dimension of the elements of the sets
    Returns:
    X -- placeholder for the stacked sets
    Y -- placeholder for the response variables
    segment_ids -- placeholder for the set ids
    """

    X = tf.placeholder(tf.float32, shape=[None, dim])
    y = tf.placeholder(tf.float32, shape=[None])
    segment_ids = tf.placeholder(tf.int32, shape = [None])
    return X, y, segment_ids

def shuffle(feats):
    """Permutes the sets and labels
    Arguments:
    feats -- collection containing sets and labels
    Returns :
    shuffled_feats -- named tuple containing permuted sets and labels
    """
    indexShuffle = np.random.permutation(len(feats.sets))
    feats_shuffled = [feats.sets[i] for i in indexShuffle]
    labels_shuffled = [feats.labels[i] for i in indexShuffle]
    shuffled_feats = Features(feats_shuffled, labels = labels_shuffled)
    return shuffled_feats

def model(train, validation, learning_rate = 0.0001, 
    num_epochs = 20, minibatch_size = 10):
    """Implements a six layer neural network model based on the Deep Sets architecture
    Arguments:
    train --  named tuple containing sets and labels for training set
    validation --named tuple containing sets and labels for validation set
    learning_rate -- learning rate for the optimization
    num_epochs -- number of iterations for the optimization loop
    minibatch_size -- size of minibatch (number of sets)
    seed -- set seed for reproducibility
    Returns:
    weights -- weights learnt by the model
    """

    #Read dimensions of data
    dim =  train.sets[0].shape[1]
    num_bags_train = len(train.sets)
    num_bags_validation = len(validation.sets)
    #Create placeholders
    X, y, segment_ids = create_placeholders(dim)

    #Weight initializations
    weights = init_all_weights(dim)
    # Forward propagation
    yhat = forwardprop(X, weights, segment_ids)
    loss = tf.reduce_sum(tf.square(tf.subtract(y, yhat)))
    mse =  tf.reduce_mean(tf.square(tf.subtract(y, yhat)))

    updates = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    variables = tf.global_variables()

    # Save the model
    saver = tf.train.Saver(variables, max_to_keep = 1)

    # Start the session
    sess =tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    #Training
    loss_vec = []
    validation_vec = []
    for epoch in range(num_epochs):
        train = shuffle(train)    
        num_batches = int(num_bags_train/minibatch_size)
        for i in range(num_batches):
            ini = minibatch_size*i
            end = minibatch_size*(i+1)
            if i==(num_batches-1): 
                end = num_bags_train
            batch_X = train.sets[ini:end]
            batch_y = train.labels[ini:end]
            segment_id_train = (np.hstack([x*np.ones(batch_X[x].shape[0]) 
                                for x in range(len(batch_X))]))
            batch_X_stacked = np.vstack(batch_X)
            sess.run(updates, feed_dict={X: batch_X_stacked, 
                y: batch_y, segment_ids: segment_id_train})
            if i==0:
                loss_value = sess.run(loss, feed_dict = {X: batch_X_stacked, 
                            y: batch_y, segment_ids: segment_id_train})
                print("Epoch %d ,loss: %0.4f" % (epoch + 1, loss_value))
                loss_vec.append(loss_value)
        #Validation
        segment_id_val = np.hstack([x*np.ones(validation.sets[x].shape[0]) 
                        for x in range(num_bags_validation)])
        mean_squared_error = sess.run(mse, feed_dict = {X:np.vstack(validation.sets), 
                            y: validation.labels, segment_ids:segment_id_val})
        validation_vec.append(mean_squared_error)
        print("val MSE = %0.4f" % mean_squared_error)

    #Plot loss and validation error
    plt.plot(range(num_epochs), loss_vec)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(range(num_epochs), validation_vec)
    plt.xlabel("Epoch")
    plt.ylabel("Validation error")
    plt.show()
    weights = sess.run(weights)
    return weights

def predict(test, weights):
    """Predicts response variable for dataset of features
    Arguments:
    test -- named tuple containing sets and labels for test set
    weights -- weights learnt by the model
    Returns:
    prediction -- predicted variables for test set 
    """
    W1 = tf.convert_to_tensor(weights["W1"])
    b1 = tf.convert_to_tensor(weights["b1"])
    W2 = tf.convert_to_tensor(weights["W2"])
    b2 = tf.convert_to_tensor(weights["b2"])
    W3 = tf.convert_to_tensor(weights["W3"])
    b3 = tf.convert_to_tensor(weights["b3"])
    W4 = tf.convert_to_tensor(weights["W4"])
    b4 = tf.convert_to_tensor(weights["b4"])
    W5 = tf.convert_to_tensor(weights["W5"])
    b5 = tf.convert_to_tensor(weights["b5"])
    W6 = tf.convert_to_tensor(weights["W6"])
    b6 = tf.convert_to_tensor(weights["b6"])
    num_bags_test = len(test.sets)
    dim =  test.sets[0].shape[1]
    #Create placeholders
    X, y, segment_ids = create_placeholders(dim)
    yhat = forwardprop(X, weights, segment_ids)
    segment_id_test = np.hstack([x*np.ones(test.sets[x].shape[0]) 
                for x in range(num_bags_test)])
    sess = tf.Session()
    prediction = sess.run(yhat, feed_dict = {X:np.vstack(test.sets), y: test.labels, 
                                            segment_ids:segment_id_test})
    return(prediction)

def eval_preds(y_pred, y_obs):
    """Calculates mean squared error and plots results
    Arguments:
    y_pred -- vector containing predicted response variables
    y_obs -- vector containing observed response variables
    """
    mse = np.mean(y_pred-y_obs)
    print("test mse: %0.4f" % mse)
    plt.scatter(y_pred, y_obs)
    plt.show()
    return

