# %%

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
# import torchtoolbox.transform as transforms
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import numpy as np
import gc
import os
# import cv2
import time
import datetime
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_pytorch import EfficientNet
# %matplotlib inline

# %%
#https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords#Step-2:-Data-Augmentation
#https://www.kaggle.com/code/nroman/melanoma-pytorch-starter-efficientnet/notebook
# DEVICE = "GPU"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("cdeotte/jpeg-melanoma-1024x1024")

print("Path to dataset files:", path)

# %%
train_df = pd.read_csv(os.path.join(path, 'train.csv'))
test_df =  pd.read_csv(os.path.join(path, 'test.csv'))


train_df = train_df[:1000]
test_df = test_df

# %%

class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None, meta_features = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features
        
    def __getitem__(self, index):
        try:
            im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
            if not os.path.exists(im_path):
                raise FileNotFoundError(f"Image not found: {im_path}")
            # x = cv2.imread(im_path)
            # x = PIL.Image(im_path)
            # image_path: str = self.images['ALL'][idx]
            # Open image in grayscale mode (L) with PIL
            """Going to try and use PIL"""
            x = Image.open(im_path) #.convert('L')
            
            # print(f"Opened PIL image {im_path}")
            meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

            if self.transforms:
                x = self.transforms(x)
                
            if self.train:
                y = self.df.iloc[index]['target']
                return (x, meta), y
            else:
                return (x, meta)
        except Exception as e:
            print(f"Error loading image at index {index}: {str(e)}")
            # Return a dummy sample
            x = Image.new('RGB', (256, 256))
            meta = np.zeros(len(self.meta_features), dtype=np.float32)
            if self.train:
                return (x, meta), 0
            return (x, meta)        
    # def display_image_index(self, index):
    #     im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
    #     # x = cv2.imread(im_path)
    #     # x = PIL.Image(im_path)
    #     # image_path: str = self.images['ALL'][idx]
    #     # Open image in grayscale mode (L) with PIL
    #     """Going to try and use PIL"""
    #     x = Image.open(im_path) #.convert('L')     
    #     # display(x)
   
    
    def __len__(self):
        return len(self.df)
    
class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super().__init__()
        self.arch = arch
        """ResNet with a linear layer???"""
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=512, out_features=500, bias=True)
        """EfficientNet with a linear layer??"""
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        """Meta linear layers """
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(500 + 250, 1)
        
    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output

train_transform = transforms.Compose([
    # AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs'),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Microscope(p=0.5),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# %%
# One-hot encoding of anatom_site_general_challenge feature
concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)
dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)

# Sex features
train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
train_df['sex'] = train_df['sex'].fillna(-1)
test_df['sex'] = test_df['sex'].fillna(-1) 

# Age features
train_df['age_approx'] /= train_df['age_approx'].max()
test_df['age_approx'] /= test_df['age_approx'].max()
train_df['age_approx'] = train_df['age_approx'].fillna(0)
test_df['age_approx'] = test_df['age_approx'].fillna(0)

train_df['patient_id'] = train_df['patient_id'].fillna(0)

# %%
meta_features = ['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]
meta_features.remove('anatom_site_general_challenge')

test = MelanomaDataset(df=test_df,
                       imfolder=os.path.join(path, 'test'), 
                       train=False,
                       transforms=train_transform,  # For TTA
                       meta_features=meta_features)

arch = EfficientNet.from_pretrained('efficientnet-b1')

NUM_EPOCHS = 2
ES_PATIENCE = 3
TTA = 3

# Initialize predictions
oof = np.zeros((len(train_df), 1))
preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)

# Enable mixed precision training
scaler = torch.amp.GradScaler('cuda')

skf = KFold(n_splits=2, shuffle=True, random_state=47)
if __name__ == '__main__':
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target']), 1):
        print('=' * 20, 'Fold', fold, '=' * 20)
        
        torch.cuda.empty_cache()
        
        model_path = f'model_{fold}.pth'
        best_val = 0
        patience = ES_PATIENCE
        
        # Initialize model
        arch = EfficientNet.from_pretrained('efficientnet-b1')
        model = Net(arch=arch, n_meta_features=len(meta_features))
        model = model.to(device)
        
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.2)
        criterion = nn.BCEWithLogitsLoss()
        
        # Create datasets and loaders (unchanged)
        train = MelanomaDataset(df=train_df.iloc[train_idx].reset_index(drop=True), 
                            imfolder=os.path.join(path, 'train'), 
                            train=True, 
                            transforms=train_transform,
                            meta_features=meta_features)
        val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True), 
                                imfolder=os.path.join(path, 'train'), 
                                train=True, 
                                transforms=test_transform,
                                meta_features=meta_features)    
        
        train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(dataset=val, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        
        for epoch in range(NUM_EPOCHS):
            print(f"epoch {epoch} started")
            start_time = time.time()
            correct = 0
            epoch_loss = 0
            model.train()
            
            # Training loop
            for i, (x, y) in enumerate(train_loader):
                # Transfer to GPU and ensure correct types
                x[0] = x[0].to(device, non_blocking=True)
                x[1] = x[1].to(device, non_blocking=True)
                y = y.float().to(device, non_blocking=True)  # Ensure y is float
                
                # Clear gradients
                optim.zero_grad(set_to_none=True)
                
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                # with torch.cuda.amp.autocast(True):
                    z = model(x)
                    loss = criterion(z, y.view(-1, 1))  # Use view instead of unsqueeze
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                
                # Calculate metrics
                with torch.no_grad():
                    pred = torch.round(torch.sigmoid(z))
                    correct += (pred == y.view(-1, 1)).sum().item()
                    epoch_loss += loss.item()
                
                if i % 10 == 0:
                    torch.cuda.empty_cache()
            
            train_acc = correct / len(train_idx)
            
            # Validation loop
            print("Eval mode...")
            model.eval()
            val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
            
            with torch.no_grad():
                for j, (x_val, y_val) in enumerate(val_loader):
                    x_val[0] = x_val[0].to(device, non_blocking=True)
                    x_val[1] = x_val[1].to(device, non_blocking=True)
                    y_val = y_val.float().to(device, non_blocking=True)
                    
                    z_val = model(x_val)
                    val_pred = torch.sigmoid(z_val)
                    val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
                
                val_acc = accuracy_score(train_df.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()).numpy())
                val_roc = roc_auc_score(train_df.iloc[val_idx]['target'].values, val_preds.cpu().numpy())
                
                print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
                    epoch + 1, epoch_loss, train_acc, val_acc, val_roc,
                    str(datetime.timedelta(seconds=time.time() - start_time))[:7]))
                
                scheduler.step(val_roc)
                
                if val_roc >= best_val:
                    best_val = val_roc
                    patience = ES_PATIENCE
                    torch.save(model.state_dict(), model_path)
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        break
            
            torch.cuda.empty_cache()
        
        # Load best model for predictions
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Final validation predictions
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = x_val[0].to(device, non_blocking=True)
                x_val[1] = x_val[1].to(device, non_blocking=True)
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()
            
            # Test predictions with TTA
            tta_preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)
            for _ in range(TTA):
                for i, x_test in enumerate(test_loader):
                    x_test[0] = x_test[0].to(device, non_blocking=True)
                    x_test[1] = x_test[1].to(device, non_blocking=True)
                    z_test = model(x_test)
                    z_test = torch.sigmoid(z_test)
                    tta_preds[i*test_loader.batch_size:i*test_loader.batch_size + x_test[0].shape[0]] += z_test
            preds += tta_preds / TTA
            
            torch.cuda.empty_cache()

    preds /= skf.n_splits
    sns.kdeplot(pd.Series(preds.cpu().numpy().reshape(-1,)));

# %%
#Kernel Density Estimation. 
#Essentially just a histogram that is continuous



# %%
# # First, verify your data paths and DataFrame
# print("DataFrame info:")
# print(train_df.info())
# print("\nMeta features:", meta_features)
# print("\nFirst few rows of DataFrame:")
# print(train_df.head())

# # Create minimal transforms
# train_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # Create dataset with error checking
# try:
#     train_dataset = MelanomaDataset(
#         df=train_df,
#         imfolder=os.path.join(path, 'train'),  # Verify this path!
#         train=True,
#         transforms=train_transform,
#         meta_features=meta_features
#     )
#     print("\nDataset created successfully")
# except Exception as e:
#     print(f"Failed to create dataset: {str(e)}")
#     raise

# # Test single sample loading
# print("\nTesting single sample loading:")
# try:
#     sample = train_dataset[0]
#     print("Successfully loaded first sample")
#     print(f"Sample types: {type(sample[0][0])}, {type(sample[0][1])}, {type(sample[1])}")
# except Exception as e:
#     print(f"Failed to load sample: {str(e)}")
#     raise

# # Create and test DataLoader with minimal settings
# try:
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=4,
#         shuffle=False,
#         num_workers=0,  # Set to 0 first to test without multiprocessing
#         pin_memory=True
#     )
#     print("\nDataLoader created successfully")
# except Exception as e:
#     print(f"Failed to create DataLoader: {str(e)}")
#     raise

# # Test batch loading
# print("\nTesting batch loading:")
# try:
#     first_batch = next(iter(train_loader))
#     print("Successfully loaded first batch")
#     print(f"Batch types: {type(first_batch[0][0])}, {type(first_batch[0][1])}, {type(first_batch[1])}")
# except Exception as e:
#     print(f"Failed to load batch: {str(e)}")
#     raise

# %%



