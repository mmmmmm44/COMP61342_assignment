# bash script to run SIFT training (python) on both datasets
# Usage: bash bash_script_sift.sh

# Set the path to the SIFT executable
CNN_COCO_PATH='model_2_cnn_reduced_coco.py'
CNN_ICUB_PATH='model_2_cnn_icub.py'

# check whether both paths exist
if [ ! -f "$CNN_COCO_PATH" ]; then
    echo "Error: $CNN_COCO_PATH not found."
    exit 1
fi
if [ ! -f "$CNN_ICUB_PATH" ]; then
    echo "Error: $CNN_ICUB_PATH not found."
    exit 1
fi

# execute
# we first train efficientnet-B0 and resnet50 on COCO dataset
# then we train the same models on iCub dataset

# printf "Running CNN training on COCO dataset..."
# printf "On EfficientNet-B0..."
# python $CNN_COCO_PATH --model_type efficientnet-b0
# printf "EfficientNet-B0 training completed."
# printf  "\n"
# printf "On ResNet50..."
# python $CNN_COCO_PATH --model_type resnet50
# printf "ResNet50 training completed."
# printf  "\n"
# printf "On EfficientNet-B4..."
# python $CNN_COCO_PATH --model_type efficientnet-b4
# printf "EfficientNet-B4 training completed."
# printf  "\n"

printf "\n\n"

printf "Running CNN training on iCub dataset..."
printf "On EfficientNet-B0..."
python $CNN_ICUB_PATH --model_type efficientnet-b0
printf "EfficientNet-B0 training completed."
printf "\n"
printf "On ResNet50..."
python $CNN_ICUB_PATH --model_type resnet50
printf "ResNet50 training completed."
printf "\n"
printf "On EfficientNet-B4..."
python $CNN_ICUB_PATH --model_type efficientnet-b4
printf "EfficientNet-B4 training completed."
printf "\n"

printf "Training completed.\n"

