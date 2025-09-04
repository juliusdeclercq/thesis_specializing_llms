#!/bin/bash

# Define the years
years=$(seq 2024 2024)

# Base directory
BASE_DIR="/projects/prjs1109/data/raw"

# Define the specific filenames to copy
FILES_TO_COPY=("20241231" "20241230" "20241227" "20241226" "20241223")

# Loop over the years
for YEAR in $years; do
    # Define source and destination directories
    SRC_DIR="${BASE_DIR}/${YEAR}"
    DEST_DIR="${BASE_DIR}/${YEAR}_RETRY"

    # Create the destination directory if it doesn't exist
    mkdir -p "$DEST_DIR"

    # Copy the specified files if they exist
    for FILE in "${FILES_TO_COPY[@]}"; do
        if [[ -f "${SRC_DIR}/${FILE}.nc.tar.gz" ]]; then
            cp "${SRC_DIR}/${FILE}.nc.tar.gz" "$DEST_DIR/"
            echo "Copied ${SRC_DIR}/${FILE} to $DEST_DIR/"
        else
            echo "File ${SRC_DIR}/${FILE} not found, skipping."
        fi
    done
done
