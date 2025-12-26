#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# URLs for MNIST dataset
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
TRAIN_IMAGES="$BASE_URL/train-images-idx3-ubyte.gz"
TRAIN_LABELS="$BASE_URL/train-labels-idx1-ubyte.gz"
TEST_IMAGES="$BASE_URL/t10k-images-idx3-ubyte.gz"
TEST_LABELS="$BASE_URL/t10k-labels-idx1-ubyte.gz"

echo "Downloading MNIST dataset..."

# Download files
curl -L -o data/train-images-idx3-ubyte.gz $TRAIN_IMAGES
curl -L -o data/train-labels-idx1-ubyte.gz $TRAIN_LABELS
curl -L -o data/t10k-images-idx3-ubyte.gz $TEST_IMAGES
curl -L -o data/t10k-labels-idx1-ubyte.gz $TEST_LABELS

echo "Download complete. Files saved in the 'data' directory."