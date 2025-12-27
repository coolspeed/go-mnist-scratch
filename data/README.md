# MNIST Dataset

This directory contains the MNIST dataset files used for training and testing the neural network.

## Files

- **`train-images-idx3-ubyte.gz`**: Training set images (60,000 samples)
- **`train-labels-idx1-ubyte.gz`**: Training set labels (60,000 samples)
- **`t10k-images-idx3-ubyte.gz`**: Test set images (10,000 samples)
- **`t10k-labels-idx1-ubyte.gz`**: Test set labels (10,000 samples)

## Format

The files are stored in the IDX file format, a simple format for vectors and multidimensional matrices of various numerical types. They are compressed using gzip.

- **Images**: 28x28 grayscale pixels, normalized in the code.
- **Labels**: Integer values from 0 to 9 representing the digit.

## Source

The dataset is downloaded from: https://ossci-datasets.s3.amazonaws.com/mnist
Original source: [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)

## Setup

These files are automatically downloaded by running the `download_data.sh` script in the root directory.
