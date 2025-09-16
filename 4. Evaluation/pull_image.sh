#!/bin/bash
set -euo pipefail   # Making sure the script halts at an error.




#################################################################################
############################### SET IMAGE ADDRESS ###############################
#################################################################################
 
# Also used this script to try to build NVIDIA's tensorrt-llm image, but that failed through singularity/apptainer, which is the only available alternative to Docker on Snellius.  
 
IMAGE_ADDRESS="docker://nvcr.io/tothemoon/pixiu:latest 
IMAGE_NAME="PIXIU"

# IMAGE_ADDRESS="docker://nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc2" # tensorrt-llm, rc2 is newest 
# IMAGE_NAME="tensorrt-llm"

#################################################################################
############################## SET TEMP DIR #####################################
#################################################################################

SCRATCH=/scratch-shared/$USER
export APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp0
mkdir -p $APPTAINER_TMPDIR
rm -r $APPTAINER_TMPDIR     # Make sure the temp dir is cleaned out.
mkdir -p $APPTAINER_TMPDIR


#################################################################################
################################ EXECUTE ########################################
#################################################################################
echo -e "\nStarting build...\n"

# Make sure that API_keys.json is in the thesis dir.
key=$(jq -r '.NGC' "$HOME/thesis/API_keys.json")    

# singularity = apptainer
apptainer registry login --username '$oauthtoken' --password $key docker://nvcr.io   # Change to oras:// if docker:// fails.


apptainer pull ${IMAGE_NAME}.sif $IMAGE_ADDRESS     # This doesn't work for tensorrt-llm somehow. 


# apptainer build --sandbox $IMAGE_NAME $IMAGE_ADDRESS     # This is sandbox mode, which allows skipping the SIF image build. 


