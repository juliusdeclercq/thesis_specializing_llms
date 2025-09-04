#!/bin/bash

# Define the years
years=$(seq 2020 2024)

# Base directory
BASE_DIR="/projects/prjs1109/data/raw"

# Loop over the years
for YEAR in $years; do
    # Define source and destination directories
    SRC_DIR="${BASE_DIR}/${YEAR}"
    DEST_DIR="${BASE_DIR}/${YEAR}_LARGE"

    # Create the destination directory if it doesn't exist
    mkdir -p "$DEST_DIR"

    # Move files larger than 3GB
    find "$SRC_DIR" -type f -size +3G -exec mv {} "$DEST_DIR" \;

    echo "Moved files larger than 3GB from $SRC_DIR to $DEST_DIR"
done
