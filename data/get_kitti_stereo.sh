#!/bin/bash

set -eu

readonly DATA_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DATA_DIR

readonly TMP_FILE=$(mktemp /tmp/tmp.kitti_stereo.XXXXXXXXXX)

readonly URL=https://s3-us-west-2.amazonaws.com/krasin-tmp/data_stereo_flow.zip
readonly DEST_DIR=kitti_stereo

if [ -d "$DEST_DIR" ]; then
    echo "Destination directory $DEST_DIR already exists. The dataset is likely already downloaded."
    echo "Abort."
    exit 1
fi

wget -O $TMP_FILE $URL
mkdir $DEST_DIR
cd $DEST_DIR
unzip $TMP_FILE

echo "Dataset has been downloaded into ${DATA_DIR}/${DEST_DIR}"
