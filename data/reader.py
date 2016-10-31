import gzip
import numpy as np

def load_mnist(base='./data/mnist'):
    """
    load_mnist taken from https://github.com/Lasagne/Lasagne/blob/master/examples/images.py
    Make sure to run get_data.sh first to download data.
    :param base: base path to images dataset
    """

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # We can now download and read the training and test set image and labels.
    X_train = load_mnist_images(base + '/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(base + '/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(base + '/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(base + '/t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    X_val, y_val = X_test, y_test

    return (X_train, y_train, X_val, y_val, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 1, 28, 28)
