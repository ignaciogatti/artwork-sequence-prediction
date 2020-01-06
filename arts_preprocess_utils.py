import h5py
import numpy as np
import math
import cv2  # for image processing

def get_image(path, img_Width=128, img_Height=128):
    #load image
    image = cv2.imread(path)
    image = cv2.resize(image, (img_Width, img_Height), interpolation=cv2.INTER_CUBIC)
    #normalize image
    image_norm = image * (1./255)
    image_norm = np.expand_dims(image_norm, axis=0)
    
    return image_norm
    


def load_dataset(path):
    train_dataset = h5py.File(path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels
    
    test_set_x_orig = np.array(train_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(train_dataset["test_set_y"][:]) # test set labels

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def batch_generator(items, labels, batch_size):
    """
    Implement batch generator that yields items in batches of size batch_size.
    There's no need to shuffle input items, just chop them into batches.
    Remember about the last batch that can be smaller than batch_size!
    Input: any iterable (list, generator, ...). You should do `for item in items: ...`
        In case of generator you can pass through your items only once!
    Output: In output yield each batch as a list of items.
    """
    
    m = items.shape[0]
    #Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/ batch_size) # number of mini batches of size mini_batch_size in your

    for i in np.arange(0,num_complete_minibatches):
        batch_items = items[i*batch_size : i*batch_size + batch_size]/255
        batch_labels = labels[i*batch_size : i*batch_size + batch_size]/255
        yield batch_items, batch_labels
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        batch_items = items[i*num_complete_minibatches:m]/255
        batch_labels = labels[i*num_complete_minibatches:m]/255
        yield batch_items, batch_labels
     

    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]/255
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]/255
        yield mini_batch_X, mini_batch_Y
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]/255
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]/255
        yield mini_batch_X, mini_batch_Y
 
    