import numpy as np

def import_data(feature_list):

    train_x = np.loadtxt("../data/multi-class-classification/train_x.txt")
    test_x = np.loadtxt("../data/multi-class-classification/test_x.txt")
    train_y = np.loadtxt("../data/multi-class-classification/train_y.txt")
    test_y = np.loadtxt("../data/multi-class-classification/test_y.txt")

    # remove uneeded features. 
    string = np.array(feature_list)
    indices = string[:] == 1

    return train_x[:, indices], test_x[:, indices], train_y, test_y
