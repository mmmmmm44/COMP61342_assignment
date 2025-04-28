# !/bin/bash
# bash script to run SIFT training (python) on both datasets
# Usage: bash bash_script_sift.sh

# Set the path to the SIFT executable
SIFT_COCO_PATH='model_1_pca_sift_coco_scriptver_v3.py'
SIFT_ICUB_PATH='model_1_pca_sift_icub_scriptver.py'

# check whether both paths exist
if [ ! -f "$SIFT_COCO_PATH" ]; then
    echo "Error: $SIFT_COCO_PATH not found."
    exit 1
fi
if [ ! -f "$SIFT_ICUB_PATH" ]; then
    echo "Error: $SIFT_ICUB_PATH not found."
    exit 1
fi

# execute
printf "Running SIFT training on COCO dataset..."
python $SIFT_COCO_PATH
printf "\n\n"

printf "Running SIFT training on iCub dataset..."
python $SIFT_ICUB_PATH
printf "\n\n"

printf "Training completed.\n"

