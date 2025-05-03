# COMP61342 Cognitive Robotics and Computer Vision Coursework

Deadline: Friday 9th May 2025, 18:00 (Blackboard)

## File Structure

```bash
|-- dataset
|   |-- coco             # dataset for original coco
|   |-- iCUB             # dataset for original iCUBWorld Transformation
|-- images               # selected result graphs/images for the report.
|-- models_coco          # trained models for self-created small COCO dataset
|-- models_icub          # trained models for self-created iCUB dataset
|-- rough_works          # past rough works at the beginning of the project
```

## For GTAs

Since there are lots of files in this repo, I will highlight the files that contribute to the coursework and explain their purposes

```bash
|-- bash_script_cnn.sh                              # bash script for executing the .py scripts on both datasets at once
|-- bash_script_sift.sh                             # bash script for executing the .py scripts on both datasets at once
|-- cnn_model_loader.py                             # code for loading pre-trained models from pytorch
|-- coco_eda.ipynb                                  # Exploratory Data Analysis (EDA) on original COCO2017 dataset
|-- early_stopper.py                                # Early Stopper for CNN fine-tuning
|-- helper_evaluations.py                           # Methods for creating different metrics (weighted acc, weighted F1-score, confusion matrix)
|-- helper.py                                       # print time-stamped log
|-- icub_eda.ipynb                                  # EDA on original iCUBWorld Transformation dataset
|-- model_1_pca_sift_coco_result.ipynb              # Jupyter notebook to analyse the results of hyperparameter search on the SIFT-based pipeline on COCO sub-dataset
|-- model_1_pca_sift_coco_scriptver_v3.py           # Python script for hyperparameter search on SIFT-based pipeline on COCO sub-dataset
|-- model_1_pca_sift_coco.ipynb                     # Notebook ver of the model_1_pca_sift_coco_scriptver_v3.py, without hyperparameter search.
|-- model_1_pca_sift_icub_result.ipynb              # Jupyter notebook to analyse the results of hyperparameter search on the SIFT-based pipeline on iCUB sub-dataset
|-- model_1_pca_sift_icub_scriptver.py              # Python script for hyperparameter search on SIFT-based pipeline on iCUB sub-dataset
|-- model_1_pca_sift_icub.ipynb                     # Notebook ver of the model_1_pca_sift_icub_scriptver.py, with hyperparameter search.
|-- model_2_cnn_icub_result.ipynb                   # Jupyter notebook to analyse the results of hyperparameter search on the CNN on iCUB sub-dataset
|-- model_2_cnn_icub.ipynb                          # Notebook ver of the model_2_cnn_icub.py, with hyperparameter search.
|-- model_2_cnn_icub.py                             # Python script for hyperparameter search on CNN on iCUB sub-dataset
|-- model_2_cnn_reduced_coco.py                     # Python script for hyperparameter search on CNN on COCO sub-dataset
|-- model_2_cnn_reduced_coco_cnn.ipynb              # Notebook ver of the model_2_cnn_reduced_coco.py, with hyperparameter search.
|-- model_2_cnn_reduced_coco_result.ipynb           # Jupyter notebook to analyse the results of hyperparameter search on the CNN on COCO sub-dataset
|-- reduced_coco_dataset_construction.ipynb         # Jupyter notebook to construct a small 10-class training & validation dataset from COCO
|-- reduced_coco_dataset_eda.ipynb                  # EDA on the created small COCO training/validation dataset
|-- reduced_icub_dataset_construction.ipynb         # Jupyter notebook to construct a small dataset from iCUBWorld Transformation
```

For the pipeline of the traditional CV model, please have a look at _model_1_pca_sift_coco_scriptver_v3.py_ and _model_1_pca_sift_icub_scriptver.py_. There should be a line staring with a for loop `for i, (K, PCA_N_COMPONENT) in enumerate(hyperparam_comb):` which describes the hyperparameter selection and the pipeline inside the training loop.

For ther results of the traditional CV model, please have a look at either _model_1_pca_sift_coco_result.ipynb_ or _model_1_pca_sift_icub_result.ipynb_ to see how the results and related graphs are generated.

For pipelines of the deep-learning based model (CNN), please have a look at _model_2_cnn_reduced_coco.py_ and _model_2_cnn_icub.py_ for the fine-tuning pipeline.

For results of the deep-learning based model (CNN), please have a look at _model_2_cnn_reducedcoco_result.ipynb_ and _model_2_cnn_icub_result.ipynb_ to find out how the graphs are generated.

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
