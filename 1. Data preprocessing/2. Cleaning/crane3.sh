#!/bin/bash

# This script moves the largest raw TAR-GZ files to a separate directory such that they do not ruin the cleaning process, because their size forms a big bottleneck causing OOM errors. Better to clean them separately. This was only found to be necessary for 2024.

# Base directory
BASE_DIR="/projects/prjs1109/data/raw"

# Define the range of years for which to move the large files. Note that (seq 2024 2024) only yields 2024, while (seq 2024) gives the range of 1 to 2024... found out the hard way.
years=$(seq 2024 2024)

# Loop over the years
for YEAR in $years; do
    # Define source and destination directories
    SRC_DIR="${BASE_DIR}/${YEAR}"
    DEST_DIR="${BASE_DIR}/${YEAR}_RETRY"

    # Create the destination directory if it doesn't exist
    mkdir -p "$DEST_DIR"

    # Find and move all files larger than 3GB
    find "$SRC_DIR" -type f -size +3G -exec mv {} "$DEST_DIR/" \;

    echo "Moved files larger than 3GB from $SRC_DIR to $DEST_DIR"
done
