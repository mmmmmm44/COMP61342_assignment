# %% [markdown]
# Using pre-trained image classification models from pytorch  
# Fine-tuned on our reduced coco-dataset (for fair comparison)
# 
# Reference: 
# 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# %%
from pycocotools.coco import COCO
import numpy as np
from pathlib import Path

from helper import print_log

# %%
train2017 = 'train2017'
val2017 = 'val2017'
ann_file = 'dataset/coco/annotations/instances_{}.json'

# %%
TOP_10_CATS_ID = set([1,  3, 62, 84, 44, 47, 67, 51, 10, 31])
CATS_NAMES = {
    1: 'person',
    3: 'car',
    62: 'chair',
    84: 'book',
    44: 'bottle',
    47: 'cup',
    67: 'dinning table',
    51: 'traffic light',
    10: 'bowl',
    31: 'handbag'
}
LABEL_LOGITS_MAPPING = {
    1: 0,
    3: 1,
    62: 2,
    84: 3,
    44: 4,
    47: 5,
    67: 6,
    51: 7,
    10: 8,
    31: 9
}
LOGITS_LABEL_MAPPING = {v:k for k, v in LABEL_LOGITS_MAPPING.items()}
LABELS = [CATS_NAMES[k] for k in LABEL_LOGITS_MAPPING.keys()]

# %%
coco_train = COCO(ann_file.format(train2017))
coco_val = COCO(ann_file.format(val2017))

# %%
def get_coco_images_and_labels(coco):

    # get all filenames
    img_ids_w_filename = {coco.dataset['images'][i]['id']: coco.dataset['images'][i]['file_name'] for i in range(len(coco.dataset['images']))}      # use dictionary for faster query

    # get all images
    img_ids = [coco.dataset['images'][i]['id'] for i in range(len(coco.dataset['images']))]

    # load labels for each imgs (as one img may have multiple labels)
    labels_per_imgs = []
    for i in range(len(img_ids)):
        labels_per_imgs.append(coco.loadAnns(coco.getAnnIds(imgIds=img_ids[i])))

    img_id_w_bb = []
    label_per_obj = []

    for labels in labels_per_imgs:
        for l in labels:
            img_id_w_bb.append((l['id'], l['image_id'], l['bbox']))
            label_per_obj.append(l['category_id'])

    return img_ids_w_filename, img_id_w_bb, label_per_obj

# %%
img_ids_w_filename_train, img_id_w_bb_train, label_per_obj_train = get_coco_images_and_labels(coco_train)
img_ids_w_filename_val, img_id_w_bb_val, label_per_obj_val = get_coco_images_and_labels(coco_val)

# %% [markdown]
# ---

# %% [markdown]
# Dataset save/load

# %%
# load filtered dataset

import pickle

filtered_dataset_dir = Path('dataset/coco_top10_filtered_20250423')

with open(filtered_dataset_dir / 'img_id_w_bb_train_top10_v2.pkl', 'rb') as f:
    img_id_w_bb_train_top10_filtered = pickle.load(f)
with open(filtered_dataset_dir / 'label_per_obj_train_top10_v2.pkl', 'rb') as f:
    label_per_obj_train_top10_filtered = pickle.load(f)

with open(filtered_dataset_dir/ 'img_id_w_bb_val_top10.pkl', 'rb') as f:
    img_id_w_bb_val_top10 = pickle.load(f)
with open(filtered_dataset_dir / 'label_per_obj_val_top10.pkl', 'rb') as f:
    label_per_obj_val_top10 = pickle.load(f)

# %%
# convert labels to logits
label_per_obj_train_top10_filtered_logits = np.array([LABEL_LOGITS_MAPPING[l] for l in label_per_obj_train_top10_filtered], dtype=np.int32)
label_per_obj_val_top10_logits = np.array([LABEL_LOGITS_MAPPING[l] for l in label_per_obj_val_top10], dtype=np.int32)

# %%
# label_per_obj_train_top10_filtered[:20]

# %%
# label_per_obj_train_top10_filtered_logits[:20]

# %% [markdown]
# ---

# %% [markdown]
# Create train-val-test set
# 
# Note that the split will be slightly different, due to the different strategy in handling images-to-bbox relationship

# %%
import torch
from torchvision.transforms import v2 as T
from torchvision.io import read_image

import math

class ReducedCOCODataset(torch.utils.data.Dataset):
    def __init__(self, img_id_w_bb:list, label_per_obj:list, img_ids_w_filename, coco_ds_name:str, transforms):
        self.img_id_w_bb = img_id_w_bb
        self.label_per_obj = label_per_obj
        self.img_ids_w_filename = img_ids_w_filename
        self.coco_ds_name = coco_ds_name
        self.transforms = transforms
        self.labels = LABELS

        self._def_transform = None

        assert self.coco_ds_name in ['train2017', 'val2017'], f"Invalid coco dataset name: {self.coco_ds_name}"

    def __getitem__(self, idx):
        # get the image id
        ann_id, img_id, bbox = self.img_id_w_bb[idx]
        label = self.label_per_obj[idx]

        # load image
        img_name = self.img_ids_w_filename[img_id]
        img_path = Path(f"dataset/coco/{self.coco_ds_name}/{img_name}")
        img = read_image(img_path)

        # chop the image to the bbox
        x1, y1, w, h = bbox
        x1 = int(math.floor(x1))
        y1 = int(math.floor(y1))
        x2 = int(math.floor(x1 + w)) + 1
        y2 = int(math.floor(y1 + h)) + 1
        img = img[:, y1:y2, x1:x2]
        

        # any data augmentation (?)

        # apply transforms
        img_t, label = self._def_transform(img, label)

        return img_t, label
    
    def get_details_from_id(self, idx):
        ann_id, img_id, bbox = self.img_id_w_bb[idx]
        label = self.label_per_obj[idx]

        # load image
        img_name = self.img_ids_w_filename[img_id]
        img_path = Path(f"dataset/coco/{self.coco_ds_name}/{img_name}")
        img = read_image(img_path)       

        return img_path, img, bbox, label
    
    def set_def_transform(self, transform):
        self._def_transform = transform
    
    def __len__(self):
        return len(self.img_id_w_bb)
        

# %%
# check dataset
dataset_traintest = ReducedCOCODataset(
    img_id_w_bb_train_top10_filtered,
    label_per_obj_train_top10_filtered_logits,
    img_ids_w_filename_train,
    coco_ds_name='train2017',
    transforms=None
)

# %%
# validation set

dataset_val = ReducedCOCODataset(
    img_id_w_bb_val_top10,
    label_per_obj_val_top10_logits,
    img_ids_w_filename_val,
    coco_ds_name='val2017',
    transforms=None
)
print('length of dataset = ', len(dataset_val), '\n')


# %% [markdown]
# ---

# %% [markdown]
# Model initialization

# %%
# build models

# %% [markdown]
# ---

# %% [markdown]
# Training loop with hyperparameter selection
# 
# we are looking for 
# - batch size (8, 16, 32)
# - learning rate (5e-4, 1e-4, 5e-5)
# 
#  cannot do batch size = 64 as GPU memory is not enough
# 
# Total combinations: $3 \times 3 = 9$

# %%
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
from itertools import product

BATCH_SIZE_GRID = [8, 16, 32]
LR_GRID = [1e-4, 5e-5, 1e-5]

MAX_EPOCHES = 15

hyperparam_combs = list(product(BATCH_SIZE_GRID, LR_GRID))
print('Total number of hyperparameter combinations: ', len(hyperparam_combs))

# %%
from datetime import datetime

from sklearn.model_selection import train_test_split


# %%
def collate_fn(batch):
  return tuple(zip(*batch))

def build_data_loaders(batch_size, dataset_train, dataset_test, dataset_val):
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=16,      # fixed for inference
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    return data_loader_train, data_loader_test, data_loader_valid

# %%
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
Function to train the model over one epoch.
'''
def train_one_epoch(model, criterion, optimizer, data_loader):
  
    train_loss = 0.0
    train_corrects = 0

    labels_list = []
    preds_list = []

    tqdm_bar = tqdm(data_loader, total=len(data_loader))
    for idx, data in enumerate(tqdm_bar):
        inputs, labels = data

        optimizer.zero_grad()

        # forward pass
        inputs = torch.stack(inputs, dim=0).to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.float64).type(torch.LongTensor).to(DEVICE)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        train_loss += loss_val
        train_corrects += (preds == labels).sum().item()

        labels_list.append(labels)
        preds_list.append(preds)
        
        tqdm_bar.set_description(desc=f"Training Loss: {loss_val:.3f}")

    train_loss /= len(data_loader)
    acc = float(train_corrects) / len(data_loader.dataset)
    labels_list = torch.cat(labels_list).cpu().numpy()
    preds_list = torch.cat(preds_list).cpu().numpy()
    print_log(f"Avg training Loss: {train_loss:.3f}; Accuracy: {acc:.3f}")

    return train_loss, acc, labels_list, preds_list

# %%
def evaluate(model, criterion, data_loader):
    test_loss = 0.0
    test_corrects = 0

    labels_list = []
    preds_list = []

    tqdm_bar = tqdm(data_loader, total=len(data_loader))

    for i, data in enumerate(tqdm_bar):
        inputs, labels = data

        inputs = torch.stack(inputs, dim=0).to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.float64).type(torch.LongTensor).to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            

        loss_val = loss.item()
        test_loss += loss_val
        test_corrects += (preds == labels).sum().item()

        labels_list.append(labels)
        preds_list.append(preds)

        tqdm_bar.set_description(desc=f"Testing Loss: {loss_val:.4f}")

    labels_list = torch.cat(labels_list).cpu().numpy()
    preds_list = torch.cat(preds_list).cpu().numpy()
    test_loss /= len(data_loader)
    acc = float(test_corrects) / len(data_loader.dataset)
    print_log(f"Avg testing Loss: {test_loss:.3f}; Accuracy: {acc:.3f}")
    
    return test_loss, acc, labels_list, preds_list

# %%
def evaluation_pred(model, data_loader, stage:str):
    
    labels_list = []
    preds_list =[]

    model.eval()
    
    tqdm_bar = tqdm(data_loader, total=len(data_loader))
    for i, data in enumerate(tqdm_bar):
        inputs, labels = data

        images = torch.stack(inputs, dim=0).to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.float64).type(torch.LongTensor).to(DEVICE)

        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        labels_list.append(labels)
        preds_list.append(preds)

        tqdm_bar.set_description(f"Evaluating {stage}")

    labels_list = torch.cat(labels_list).cpu().numpy()
    preds_list = torch.cat(preds_list).cpu().numpy()
    
    return labels_list, preds_list

# %%
# compute acc, confusion matrix, classification report
from helper_evaluations import compute_accuracy, compute_f1_score, compute_balanced_accuracy, compute_classification_report, compute_confusion_matrix

def compute_classification_metrics(target_labels, pred_labels, target_names:list[str]):
    
    # compute accuracy
    acc = compute_accuracy(target_labels, pred_labels)
    print("Accuracy: ", acc)

    report = compute_classification_report(target_labels, pred_labels, target_names)
    print("Classification Report:\n", report)

    # compute confusion matrix
    compute_confusion_matrix(target_labels, pred_labels, target_names, save=False)

    return target_labels, pred_labels

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
# define early stopper -> w/out the need to test epoch; just need to define the patience
# and maximum epoches for the model

# %%
from cnn_model_loader import ModelType, load_model, get_default_transforms
from early_stopper import EarlyStopper


def main(model_type:ModelType):

    # add default transforms to datasets (before creating subset and loaders)
    dataset_traintest.set_def_transform(get_default_transforms(model_type))
    dataset_val.set_def_transform(get_default_transforms(model_type))

    # create train and validation set
    train_indices, test_indices = train_test_split(list(range(len(dataset_traintest.img_id_w_bb))), test_size=0.2, random_state=42)

    dataset_train = torch.utils.data.Subset(dataset_traintest, train_indices)
    dataset_test = torch.utils.data.Subset(dataset_traintest, test_indices)

    dataset_traintest

    tdy = datetime.now()
    top_model_dir = Path(f'models_coco/{str(model_type)}/{tdy.strftime("%Y%m%d-%H%M%S")}/')
    if not top_model_dir.exists():
        top_model_dir.mkdir(parents=True)

    for batch_size, lr in hyperparam_combs:
        print_log(f'-' * 50)
        print_log(f"Batch size: {batch_size}, Learning rate: {lr}")
        print_log(f'-' * 50)

        model_dir = top_model_dir/f'bs_{batch_size}_lr_{lr}'
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
            print_log(f"Model directory {model_dir} created")
        
        model_name = f"{str(model_type)}_bs_{batch_size}_lr_{lr}"

        if Path(model_dir/model_name).exists():
            print_log(f"Model {model_dir/model_name} already exists, skipping...")
            continue
        
        # create dataloaders
        data_loader_train, data_loader_test, data_loader_valid = build_data_loaders(batch_size, dataset_train, dataset_test, dataset_val)

        # init model
        model = load_model(model_type, out_features=len(LABELS))
        model.to(DEVICE)

        # construct optimizer, learning rate scheduler etc.
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)       # reduce lr by 0.99 every epoch

        # early stopping
        # we monitor the testing accuracy
        early_stopper = EarlyStopper(patience=5, delta=0.005, minimize=False)

        loss_dict = {'train_loss': [], 'test_loss': []}
        acc_dict = {'train_acc': [], 'test_acc': []}

        # training
        for epoch in range(MAX_EPOCHES):
            model.train()
            print_log(f"Epoch {epoch+1}/{MAX_EPOCHES}")
            train_loss, train_acc, y_labels_train, y_pred_train = train_one_epoch(
                model,
                criterion,
                optimizer,
                data_loader_train,
            )
            loss_dict['train_loss'].append(train_loss)
            acc_dict['train_acc'].append(train_acc)

            lr_scheduler.step()

            model.eval()

            # run on test set for evaluation (get test set loss)
            with torch.no_grad():
                test_loss, test_acc, y_labels_test, y_pred_test = evaluate(
                    model,
                    criterion,
                    data_loader_test,
                )
                loss_dict['test_loss'].append(test_loss)
                acc_dict['test_acc'].append(test_acc)

            early_stop = early_stopper.step(test_acc, epoch, model)

            if early_stop:
                print_log(f"Early stopping at epoch {epoch+1}")
                break

        print_log(f"Best epoch: {early_stopper.best_epoch+1}; Best Acc: {early_stopper.best_loss:.3f}")
        # get the best model
        best_model = early_stopper.get_best_model()
        best_epoch = early_stopper.best_epoch

        # save the best model
        model_name = model_name + f'_epoch_{best_epoch+1}.pth'
        torch.save(best_model.state_dict(), model_dir/model_name)
        print_log(f"Model saved to {model_dir/model_name}")

        # save the loss dict
        loss_dict_path = model_dir/'loss_dict.pkl'
        with open(loss_dict_path, 'wb') as f:
            pickle.dump(loss_dict, f)
        print_log(f"Loss dict saved to {loss_dict_path}")


        # evaluate the model on train, test and validation set
        best_model.eval()

        # train set
        # save_evaluations(y_labels_train, y_pred_train, LABELS, model_dir, eval_stage='train')
        # y_labels_train_path = model_dir / 'y_labels_train.pkl'
        # y_pred_train_path = model_dir / 'y_pred_train.pkl'
        # np.save(y_labels_train_path, y_labels_train)
        # np.save(y_pred_train_path, y_pred_train)

        y_labels_test, y_pred_test = evaluation_pred(best_model, data_loader_test, stage='test')
        save_evaluations(y_labels_test, y_pred_test, LABELS, model_dir, eval_stage='test')
        y_labels_test_path = model_dir / 'y_labels_test'
        y_pred_test_path = model_dir / 'y_pred_test'
        np.save(y_labels_test_path, y_labels_test)
        np.save(y_pred_test_path, y_pred_test)

        # save the model predictions
        y_labels_valid, y_pred_valid = evaluation_pred(best_model, data_loader_valid, stage='validation')
        save_evaluations(y_labels_valid, y_pred_valid, LABELS, model_dir, eval_stage='validation')
        y_labels_valid_path = model_dir / 'y_labels_valid'
        y_pred_valid_path = model_dir / 'y_pred_valid'
        np.save(y_labels_valid_path, y_labels_valid)
        np.save(y_pred_valid_path, y_pred_valid)

        print_log(f"Finished training for batch size: {batch_size}, learning rate: {lr}")
        print_log(f'-' * 50)

if __name__ == '__main__':
    # takes an argument for the model type
    import argparse
    parser = argparse.ArgumentParser(description='Train a model on coco dataset')
    parser.add_argument('--model_type', type=str, 
                        default='efficientnet-b0', help='Model type to use',
                        choices=ModelType.available_models())
    args = parser.parse_args()

    model_type = args.model_type
    print_log(f"Model type: {model_type}")

    # to ModelType
    if model_type == ModelType.EFFICIENTNET_B0.value:
        model_type = ModelType.EFFICIENTNET_B0
    elif model_type == ModelType.EFFICIENTNET_B4.value:
        model_type = ModelType.EFFICIENTNET_B4
    elif model_type == ModelType.RESNET50.value:
        model_type = ModelType.RESNET50
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # run the main function
    main(model_type)