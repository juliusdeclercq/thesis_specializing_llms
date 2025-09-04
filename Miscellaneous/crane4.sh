#!/bin/bash

# Define the base directories
LARGE_DIR="/projects/prjs1109/intermediate/2024_RETRY"
BASE_DEST="/projects/prjs1109/intermediate"

# Define the leaf directories to process
LEAF_DIRS=("filings" "logs/filing_info" "logs/logs")

# Function to extract year from filename
extract_year() {
    local filename="$1"

    # Case 1: .tar files (year at the beginning)
    if [[ "$filename" =~ ^([0-9]{4}) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi

    # Case 2: filing_info_YYYYMMDD.jsonl
    if [[ "$filename" =~ filing_info_([0-9]{4})[0-9]{4}\.jsonl$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi

    # Case 3: logs_YYYYMMDD.pkl
    if [[ "$filename" =~ logs_([0-9]{4})[0-9]{4}\.pkl$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi

    # If no match, return a warning
    echo ""
}

# Loop through each leaf directory
for LEAF in "${LEAF_DIRS[@]}"; do
    SRC_DIR="${LARGE_DIR}/${LEAF}"
    
    # Ensure the source directory exists
    if [[ ! -d "$SRC_DIR" ]]; then
        echo "Skipping: $SRC_DIR (directory not found)"
        continue
    fi

    echo "Processing directory: $SRC_DIR"

    # Process all files in the directory
    for FILE in "$SRC_DIR"/*; do
        [ -e "$FILE" ] || continue  # Skip if no files exist

        FILENAME=$(basename "$FILE")
        YEAR=$(extract_year "$FILENAME")

        if [[ -n "$YEAR" ]]; then
            DEST_DIR="${BASE_DEST}/${YEAR}/${LEAF}"
            mkdir -p "$DEST_DIR"
            mv "$FILE" "$DEST_DIR/"
            echo "Moved $FILENAME to $DEST_DIR/"
        else
            echo "Skipping $FILENAME (no valid year found)"
        fi
    done
done
