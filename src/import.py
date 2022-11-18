import numpy as np

def import_data(feature_string):

    train_x = np.loadtxt("../data/train_x.txt")
    test_x = np.loadtxt("../data/test_x.txt")
    train_y = np.loadtxt("../data/train_y.txt")
    test_y = np.loadtxt("../data/test_y.txt")

    # remove uneeded features. 
    string = np.array(list(feature_string))
    string = string.reshape((-1, 1))
    indices = string[:, 0] == '1'

    return train_x[:, indices], test_x[:, indices], train_y, test_y

