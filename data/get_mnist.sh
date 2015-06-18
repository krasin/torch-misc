#!/bin/bash
# This script downloads MNIST dataset in Torch format.
# See also https://github.com/torch/tutorials/tree/master/A_datasets

set -ue

readonly DATA_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DATA_DIR

readonly TMP_FILE=$(mktemp /tmp/tmp.mnist.XXXXXXXXXX)

# Taken from https://github.com/torch/tutorials/blob/master/A_datasets/mnist.lua
readonly URL=http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz
readonly DEST_DIR=mnist.t7

if [ -d "$DEST_DIR" ]; then
    echo "Destination directory $DEST_DIR already exists. The dataset is likely already downloaded."
    echo "Abort."
    exit 1
fi

wget -O $TMP_FILE http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz
tar -xzf $TMP_FILE

echo "MNIST dataset has been downloaded into ${DATA_DIR}/${DEST_DIR}"

echo "Downloading custom test dataset..."
wget https://s3-us-west-2.amazonaws.com/krasin-tmp/digits.png
echo "Done"
