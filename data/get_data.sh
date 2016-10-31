#!/bin/bash

mkdir ./data/

# Download MNIST dataset
mkdir ./data/mnist
curl -o ./data/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o ./data/mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o ./data/mnist/t10k-images-idx3-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o ./data/mnist/t10k-labels-idx1-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz