# %% [markdown]
# The model training script for SIFT on modified iCUB World Transformation dataset
# 
# 20 classes, with cropped images and better cropped images from manual annotation
# 
# TODO: change it to pyscript and execute

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
import os
import pickle

from helper import print_log

# %%
dataset_dir = Path('dataset/icub_custom_subset')

# %%
cats = sorted([cat for cat in os.listdir(dataset_dir) if os.path.isdir(dataset_dir / cat)])
LABEL_TO_CAT = {i:cat for i, cat in enumerate(cats)}
CAT_TO_LABEL = {cat:i for i, cat in enumerate(cats)}
LABELS = cats
print(cats); print(); print()
print(LABEL_TO_CAT); print(); print()
print(CAT_TO_LABEL)

# %%
# load the train-test set and validation set

train_test_set_path = dataset_dir / 'train_test_set.pkl'
train_test_set_label_path = dataset_dir / 'train_test_set_labels.pkl'
eval_set_path = dataset_dir / 'eval_set.pkl'
eval_set_label_path = dataset_dir / 'eval_set_labels.pkl'

if not train_test_set_path.exists():
    raise FileNotFoundError(f"Train-test set not found at {train_test_set_path}")
if not train_test_set_label_path.exists():
    raise FileNotFoundError(f"Train-test set labels not found at {train_test_set_label_path}")
if not eval_set_path.exists():
    raise FileNotFoundError(f"Validation set not found at {eval_set_path}")


train_test_set = pickle.load(open(train_test_set_path, 'rb'))
train_test_set_label = pickle.load(open(train_test_set_label_path, 'rb'))
eval_set = pickle.load(open(eval_set_path, 'rb'))
eval_set_label = pickle.load(open(eval_set_label_path, 'rb'))

# %% [markdown]
# ---

# %% [markdown]
# Create train-test split
# 
# Training-test split has ratio 4:1; with stratify consideration

# %%
from sklearn.model_selection import train_test_split

# Split the train-test set into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    train_test_set, train_test_set_label, test_size=0.2, random_state=42, stratify=train_test_set_label
)

# %%
from tqdm import tqdm

# 1. First, prepare your data for the datasets library
def prepare_dataset_list(X):
    # Create a lightweight dictionary containing only metadata (not images)

    dataset_list = []
    
    for sample in X:
        per_img_dict = {}
        per_img_dict["image_id"] = '/'.join(sample.parts[2:])
        per_img_dict["image_path"] = sample

        dataset_list.append(per_img_dict)
        
    return dataset_list

# 2. Define the SIFT processing function

def process_image_with_sift(example):
    """Process a single image, extracting SIFT features"""
    # Load image only when needed
    img_path = example['image_path']
    img = cv2.imread(str(img_path))
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    # Convert keypoints to serializable format
    serialized_keypoints = []
    for kp in keypoints:
        serialized_keypoints.append({
            'x': float(kp.pt[0]),
            'y': float(kp.pt[1]), 
            'size': float(kp.size),
            'angle': float(kp.angle),
            'response': float(kp.response),
            'octave': int(kp.octave)
        })
    
    # Return only the features, together with the image_id, bbox, and filename
    # but not the image (pixel) itself
    return {
        'image_id': example['image_id'],
        'image_path': example['image_path'],
        'keypoints': serialized_keypoints,
        'descriptors': descriptors if descriptors is not None else np.array([])
    }

def create_sift_dataset(X):
    # Create the dataset dictionary
    dataset_list_init = prepare_dataset_list(X)

    dataset_list = []

    with tqdm(total=len(dataset_list_init)) as pbar:
        pbar.set_description("Extracting SIFT features")

        for i, example in enumerate(dataset_list_init):
            # Process using (optional) multi-processing
            processed_example = process_image_with_sift(example)
            dataset_list.append(processed_example)
            pbar.update(1)

    return dataset_list

# %%
sift_dataset_train_path = dataset_dir.joinpath('sift_dataset_train.pkl')

if sift_dataset_train_path.exists():
    sift_dataset_train = pickle.load(open(sift_dataset_train_path, 'rb'))

    print("Training dataset already exists. Loading from disk...")

else:
    # Create the dataset
    sift_dataset_train = create_sift_dataset(X_train)

    # Save the dataset to disk
    pickle.dump(sift_dataset_train, open(sift_dataset_train_path, 'wb'))

    print("Training dataset is created and saved to disk.")

# %%
sift_dataset_test_path = dataset_dir.joinpath('sift_dataset_test.pkl')

if sift_dataset_test_path.exists():
    sift_dataset_test = pickle.load(open(sift_dataset_test_path, 'rb'))

    print("Testing dataset already exists. Loading from disk...")

else:
    # Create the dataset
    sift_dataset_test = create_sift_dataset(X_test)

    # Save the dataset to disk
    pickle.dump(sift_dataset_test, open(sift_dataset_test_path, 'wb'))

    print("Testing dataset is created and saved to disk.")

# %%
sift_dataset_val_path = dataset_dir.joinpath('sift_dataset_val.pkl')
if sift_dataset_val_path.exists():
    sift_dataset_val = pickle.load(open(sift_dataset_val_path, 'rb'))

    print("Validation dataset already exists. Loading from disk...")
else:
    # create the dataset
    sift_dataset_val = create_sift_dataset(eval_set)

    # try to save the dataset
    pickle.dump(sift_dataset_val, open(sift_dataset_val_path, 'wb'))

    print("Validation dataset is created and saved to disk.")

# %%
def load_all_descriptors(sift_dataset):

    all_descriptors = []
    for example in sift_dataset:
        # check if descriptors is not empty
        if example['descriptors'].ndim < 2:
            continue

        all_descriptors.append(example['descriptors'])
    # convert to numpy array
    all_descriptors_np = np.concatenate(all_descriptors, axis=0)

    return all_descriptors_np

# %%
# grab all descriptors (write them to memory, as KMeans have no incremental fit)
all_descriptors_train_path = dataset_dir.joinpath('all_descriptors_train.npy')

if all_descriptors_train_path.exists():
    all_descriptors_np_train = np.load(all_descriptors_train_path)
    
    print("All descriptors for training dataset already exists. Loading from disk...")

else:
    all_descriptors_np_train = load_all_descriptors(sift_dataset_train)
    
    # save to disk
    np.save(all_descriptors_train_path, all_descriptors_np_train)

    print("All descriptors for training dataset is created and saved to disk.")


# %% [markdown]
# ---

# %% [markdown]
# Model

# %%
DESCIPTORS_DIM = len(sift_dataset_train[0]['descriptors'][0])

# %% [markdown]
# ---

# %%
from itertools import product

K_GRID = [20, 40, 80, 160, 320, 640]              # number of visual words
PCA_N_COMPONENTS_GRID = [20, 50, 128]       # 128 is the default for SIFT -> no PCA reduction. Also this suits the ratio in natural log

hyperparam_comb = list(product(K_GRID, PCA_N_COMPONENTS_GRID))

# %%
from datetime import datetime

tdy = datetime.now()
top_model_dir = Path(f'models_icub/PCA-SIFT/{tdy.strftime("%Y%m%d-%H%M%S")}/')
if not top_model_dir.exists():
    top_model_dir.mkdir(parents=True)

# %%
# base on today's date, set the random seed
# then randomly create random seed for the each K-mean model
import random

random.seed(int(tdy.strftime("%Y%m%d")))

# create a random seed for each K-mean model
seed_for_KMeans = [random.randint(0, 2147483647) for _ in range(len(hyperparam_comb))]   # 2^32 - 1
print(seed_for_KMeans)

# create a random seed for each RBF kernel approximation
seed_for_RBF = [random.randint(0, 2147483647) for _ in range(len(hyperparam_comb))]   # 2^32 - 1
print(seed_for_RBF)

# create a random seed for each LinearSVC model
seed_for_LinearSVC = [random.randint(0, 2147483647) for _ in range(len(hyperparam_comb))]   # 2^32 - 1
print(seed_for_LinearSVC)

# %%
from sklearn.decomposition import PCA
def PCA_training(X, n_components):
    """Apply PCA to the data"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

# %%
from sklearn.cluster import KMeans

def KMeans_training(X, n_clusters, random_state):
    """Apply KMeans to the data"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans

# %%
# evaluation test-set function

# extend the dataset with the cluster id (equivalent to vector quantization)
def _assign_cluster_id(example, pca, kmeans, pca_n_components):
    """Assign cluster id to each keypoint based on the closest cluster center"""
    # Update the example with the cluster ids

    des = np.array(example['descriptors'])
    # check if descriptors are empty
    if des.size == 0:
        example['cluster_ids'] = np.array([], dtype=np.int32)
        return example
    
    # apply PCA to the descriptors
    if pca_n_components < DESCIPTORS_DIM:
        red_des = pca.transform(des)
    else:
        red_des = des

    # early return if transformed descriptors are empty
    if red_des.size == 0:
        example['cluster_ids'] = np.array([], dtype=np.int32)
        return example

    if red_des.ndim == 1:
        example['cluster_ids'] = kmeans.predict(red_des.reshape(1, -1))
    else:
        example['cluster_ids'] = kmeans.predict(red_des)
    
    return example

def assign_cluster_id(sift_dataset, pca, kmeans, pca_n_components):
    # Apply the filter to the dataset
    with tqdm(total=len(sift_dataset)) as pbar:
        pbar.set_description("Assigning cluster ids to keypoints")
        for idx, example in enumerate(sift_dataset):
            sift_dataset[idx] = _assign_cluster_id(example, pca, kmeans, pca_n_components)
            pbar.update(1)


# %%
# create a histogram of the cluster ids
# that will be used to compute TF-IDF

def _create_histogram(example, K):
    """Create a histogram of cluster ids"""

    # early exit if descriptors are empty -> cluster_ids will be empty too
    if len(example['descriptors']) == 0:
        example['histogram'] = np.array([[]], dtype=np.int64)
        return example

    hist, _ = np.histogram(example['cluster_ids'], bins=np.arange(K + 1))
    
    example['histogram'] = hist.reshape(-1, K)

    return example

def create_histogram(sift_dataset, K):
    # Apply the histogram function to the dataset
    with tqdm(total=len(sift_dataset)) as pbar:
        pbar.set_description("Creating histogram of cluster ids")
        for idx, example in enumerate(sift_dataset):
            sift_dataset[idx] = _create_histogram(example, K)
            pbar.update(1)

# %%
from sklearn.feature_extraction.text import TfidfTransformer

def create_tfidf_matrix(sift_dataset):
    """Create a TF-IDF matrix from the histograms"""
    tfidf = TfidfTransformer()

    # grab all non-empty histograms and concat them to a very large 2D array
    histograms = np.array([example['histogram'] for idx, example in enumerate(sift_dataset) if len(example['histogram'][0]) > 0])
    # reshape
    histograms = histograms.reshape(histograms.shape[0], -1)
    
    # Compute the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(histograms)
    
    return tfidf, tfidf_matrix

# %%
def evaluate_model(sift_dataset, y, pca, kmeans, tfidf, rbf, svm, K, PCA_N_COMPONENT):
    # Apply the filter to the dataset
    assign_cluster_id(sift_dataset, pca, kmeans, PCA_N_COMPONENT)

    # Apply the histogram function to the dataset
    create_histogram(sift_dataset, K)

    des_histo = np.concatenate(
        [example['histogram'] for idx, example in enumerate(sift_dataset) if len(example['histogram'][0]) > 0],
        axis=0
    )
    des_histo = des_histo.reshape(des_histo.shape[0], -1)

    # Convert the list of descriptors to TF-IDF representation
    tfidf_matrix = tfidf.transform(des_histo)

    y_filtered = [y[i] for i, example in enumerate(sift_dataset) if len(example['histogram'][0]) > 0]
    y_filtered = np.array(y_filtered)

    # Predict the labels using the RBF + SVM model
    y_pred = svm.predict(
        rbf.transform(tfidf_matrix)
    )

    return y_filtered, y_pred

# %%
from helper_evaluations import compute_accuracy, compute_f1_score, compute_balanced_accuracy, compute_classification_report, compute_confusion_matrix

def save_evaluations(y, y_pred, labels, model_dir, eval_stage=str):
    """Save the evaluation results
    
    eval_stage: str
        The stage of the evaluation. It can be 'train', 'test' or 'val.
    """
    # Save the accuracy score
    accuracy = compute_accuracy(y, y_pred)
    f1 = compute_f1_score(y, y_pred)
    balanced_accuracy = compute_balanced_accuracy(y, y_pred)
    print_log(f"Accuracy [{eval_stage}]: {accuracy}; Weighted F1 [{eval_stage}]: {f1}; Weighted Accuracy [{eval_stage}]: {balanced_accuracy}")
    # save the scores
    with open(model_dir / f'accuracy_{eval_stage}.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Weighted F1: {f1}\n")
        f.write(f"Weighted Accuracy: {balanced_accuracy}\n")

    # Save the classification report
    report = compute_classification_report(y, y_pred, labels)
    with open(model_dir / f'classification_report_{eval_stage}.txt', 'w') as f:
        f.write(report)

    # Save the confusion matrix
    cm_path = model_dir / f'confusion_matrix_{eval_stage}.png'
    compute_confusion_matrix(y, y_pred, labels, save=True, save_path=cm_path)

# %%
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler

for i, (K, PCA_N_COMPONENT) in enumerate(hyperparam_comb):
    print_log(f'-' * 50)
    print_log(f'K: {K}, PCA_N_COMPONENT: {PCA_N_COMPONENT}')
    print_log(f'-' * 50)

    
    # create a directory for each model
    model_dir = top_model_dir / f'KMeans_{K}_PCA_{PCA_N_COMPONENT}'
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
        print_log(f'Model directory {model_dir} created.')
    
    pca_model_name = f'PCA-SIFT_PCA_PCA-N_{PCA_N_COMPONENT}' + '.pkl'
    kmeans_model_name = f'PCA-SIFT_KMeans_PCA-N_{PCA_N_COMPONENT}_KMeans-K_{K}' + '.pkl'
    tfidf_model_name = f'PCA-SIFT_TFIDF_PCA-N_{PCA_N_COMPONENT}_KMeans-K_{K}' + '.pkl'
    rbf_model_name = f'PCA-SIFT_RBF_PCA-N_{PCA_N_COMPONENT}_KMeans-K_{K}' + '.pkl'
    svm_model_name = f'PCA-SIFT_SVM-SGD_PCA-N_{PCA_N_COMPONENT}_KMeans-K_{K}.pkl'

    # check if the SVM model already exists
    # if exists -> skip that
    if (model_dir / svm_model_name).exists():
        print_log(f'{svm_model_name} model exists. Skip this model.')
        continue
    
    pca = None

    if PCA_N_COMPONENT < DESCIPTORS_DIM:
        # apply PCA
        pca, all_descriptors_pca = PCA_training(all_descriptors_np_train, n_components=PCA_N_COMPONENT)
        # save the PCA model
        pickle.dump(pca, open(model_dir / pca_model_name, 'wb'))
        print_log(f'{pca_model_name} model saved.')
    else:
        # no PCA needed
        all_descriptors_pca = all_descriptors_np_train

    # apply KMeans
    # check if the KMeans model already exists
    if (model_dir / kmeans_model_name).exists():
        print_log(f'{kmeans_model_name} model exists. Reload the KMeans model.')
        kmeans = pickle.load(open(model_dir / kmeans_model_name, 'rb'))
    else:
        kmeans = KMeans_training(all_descriptors_pca, n_clusters=K, random_state=seed_for_KMeans[i])
        # save the KMeans model
        pickle.dump(kmeans, open(model_dir / kmeans_model_name, 'wb'))
        print_log(f'{kmeans_model_name} model saved.')

    # apply the cluster id to the dataset
    assign_cluster_id(sift_dataset_train, pca, kmeans, PCA_N_COMPONENT)
    print_log(f'Cluster ids assigned to the dataset.')

    # Apply the histogram function to the dataset
    create_histogram(sift_dataset_train, K)
    print_log(f'Histogram of cluster ids created.')
    
    # create TF-IDF BoVW representation
    tfidf, tfidf_matrix = create_tfidf_matrix(sift_dataset_train)
    # save the TF-IDF transformer
    pickle.dump(tfidf, open(model_dir / tfidf_model_name, 'wb'))
    print_log(f'{tfidf_model_name} model saved.')

    # note that there are images with no histograms
    # need to filter them out
    y_train_filtered = [y_train[i] for i, example in enumerate(sift_dataset_train) if len(example['histogram'][0]) > 0]
    y_train_filtered = np.array(y_train_filtered)

    # SVM (either SGD, or SVC). The later requires ~40min per model
    # 0425 update: use RBF kernel + LinearSVC for support to non-linearity
    # higher n_components -> closer to SVC
    # takes around 2-3 mins
    rbf = RBFSampler(gamma=1, n_components=1000, random_state=seed_for_RBF[i])
    X_features = rbf.fit_transform(tfidf_matrix)
    # use LinearSVC for faster training
    svm = LinearSVC(tol=1e-6, C=1.0, random_state=seed_for_LinearSVC[i])
    svm.fit(X_features, y_train_filtered)

    print_log(f'RBF Kernel and LinearSVC are trained with {K} clusters and PCA {PCA_N_COMPONENT} components.')
    # save the model
    pickle.dump(rbf, open(model_dir / rbf_model_name, 'wb'))
    pickle.dump(svm, open(model_dir / svm_model_name, 'wb'))
    print_log(f'{rbf_model_name} model saved.')
    print_log(f'{svm_model_name} model saved.')


    # evaluate on both train and test set
    # evaluate on train set
    y_pred_train = svm.predict(X_features)
    save_evaluations(y_train_filtered, y_pred_train, labels=LABELS, model_dir=model_dir, eval_stage='train')
    print_log(f'Evaluation on train set done.')

    # save the prediction result for future use (create further evaluation)
    y_train_filtered_path = model_dir / f'y_train_filtered.npy'
    y_pred_train_path = model_dir / f'y_pred_train.npy'
    np.save(y_train_filtered_path, y_train_filtered)
    np.save(y_pred_train_path, y_pred_train)

    
    # evaluate on test set
    y_test_filtered, y_pred_test = evaluate_model(sift_dataset_test, y_test, pca, kmeans, tfidf, rbf, svm, K, PCA_N_COMPONENT)
    save_evaluations(y_test_filtered, y_pred_test, labels=LABELS, model_dir=model_dir, eval_stage='test')
    print_log(f'Evaluation on test set done.')

    # save the prediction result for future use (create further evaluation)
    y_test_filtered_path = model_dir / f'y_test_filtered.npy'
    y_pred_test_path = model_dir / f'y_pred_test.npy'
    np.save(y_test_filtered_path, y_test_filtered)
    np.save(y_pred_test_path, y_pred_test)

    # evaluate on validation set
    y_val_filtered, y_pred_val = evaluate_model(sift_dataset_val, eval_set_label, pca, kmeans, tfidf, rbf, svm, K, PCA_N_COMPONENT)
    save_evaluations(y_val_filtered, y_pred_val, labels=LABELS,  model_dir=model_dir, eval_stage='val')
    print_log(f'Evaluation on validation set done.')

    # save the prediction result for future use (create further evaluation)
    y_val_filtered_path = model_dir / f'y_val_filtered.npy'
    y_pred_val_path = model_dir / f'y_pred_val.npy'
    np.save(y_val_filtered_path, y_val_filtered)
    np.save(y_pred_val_path, y_pred_val)

    print_log(f'Finished training and evaluation for K={K} and PCA_N_COMPONENT={PCA_N_COMPONENT}.')
    print_log(f'-' * 50)

# 