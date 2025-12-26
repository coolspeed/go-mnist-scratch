#!/bin/bash

mkdir -p data

echo "Downloading MNIST dataset..."

curl -L -o data/train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
curl -L -o data/train-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
curl -L -o data/t10k-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
curl -L -o data/t10k-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

echo "Extracting files..."

cd data
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
cd ..

echo "Download complete!"
echo "Files saved in data/ directory"
