import numpy as np
import cPickle

def load_label_names(file):
    # unpickle from file
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return np.array(dict['fine_label_names'])



def load_cifar(file):
    # unpickle from file
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    imgs_flat = np.array(dict['data'])
    #imgs = [x.reshape(3,32,32).transpose(1,2,0) for x in imgs_flat]
    labels = np.array(dict['fine_labels'])
    return imgs_flat, labels

