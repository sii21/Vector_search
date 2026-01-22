#!/bin/bash

# Download and extract the laion-small-clip dataset
# Do not download if the dataset has already been loaded

DATA_DIR="hw4/data"
DATASET_URL="https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-small-clip.tgz"
DATASET_FILE="$DATA_DIR/laion-small-clip.tgz"
VECTORS_FILE="$DATA_DIR/vectors.npy"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if dataset is already downloaded and extracted
if [ -f "$VECTORS_FILE" ]; then
    echo "Dataset already exists at $VECTORS_FILE"
    echo "Skipping download"
    exit 0
fi

# Download the dataset
echo "Downloading dataset from $DATASET_URL..."
curl -L -o "$DATASET_FILE" "$DATASET_URL"

if [ $? -ne 0 ]; then
    echo "Error downloading dataset"
    exit 1
fi

echo "Download complete. Extracting..."

# Extract the dataset
tar -xzf "$DATASET_FILE" -C "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo "Error extracting dataset"
    exit 1
fi

echo "Extraction complete"

# Remove the archive to save space
rm "$DATASET_FILE"
echo "Removed archive file"

# List extracted files
echo "Dataset files:"
ls -lh "$DATA_DIR"

echo "Dataset ready!"
