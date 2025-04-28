# COMP61342 Cognitive Robotics and Computer Vision Coursework

Deadline: Friday 9th May 2025, 18:00 (Blackboard)

## File Structure

```bash
|-- dataset
|   |-- coco        # dataset for coco

|-- models_coco          # trained models
|-- models_icub          # trained models
|-- rough_works          # past works that does not related to this coursework any more
```

## Dataset creation

1. Download the datasets from the official website. Then read [README](dataset/README.md) under the _dataset_ folder

2. Execute _reduced_coco_dataset_construction.ipynb_ and _reduced_icub_dataset_construction.ipynb_ to create the two datasets which are subsets of COCO and iCUB respectively, under the _model_ folder

## Training

There are two types of models, one is PCA-SIFT + SVM, representative of traditional CV method, and CNN-finetuning on these two datasets, representative of deep-learning method

For smooth training on all these two models on both datasets, prepare the environment by creating a conda virtual environment from _environment.yml_ (for wsl with CUDA support). Then execute _bash_script_sift.sh_ and _bash_script_cnn.sh_.

To execute on jupyter notebook, read _model_1_pca_sift_coco.ipynb_ and  _model_1_pca_sift_icub.ipynb_ for traditional CV method, and read _model_2_cnn_reducedcoco_cnn.ipynb_ and _model_2_cnn_icub.ipynb_ for CNN-based method.

## Results & Analysis

During training, lots of data are saved, including the training results (in a dictionary, or some txt files), the prediction results (in numpy files), for quick result inspection and creation of extra evaluation metrics from prediction results of both models.

Read _model_1_pca_sift_coco_result.ipynb_ and _model_1_pca_sift_icub_result.ipynb_ for traditional CV method.
