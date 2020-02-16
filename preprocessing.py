#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import random
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# In[3]:


def make_directory(train_dir):
    # create directory for each dataset
    if not os.path.exists(os.path.join(train_dir, 'train')):
        os.mkdir(os.path.join(train_dir, 'train'))
    
    if not os.path.exists(os.path.join(train_dir, 'valid')):
        os.mkdir(os.path.join(train_dir, 'train', 'valid'))
    
    labels = ['cat', 'dog']
    
    # create directory for each label
    for label in labels:
        if not os.path.exists(os.path.join(train_dir, 'train', label)):
            os.mkdir(os.path.join(train_dir, 'train', label))
            
    for label in labels:
        if not os.path.exists(os.path.join(train_dir, 'valid', label)):
            os.mkdir(os.path.join(train_dir, 'valid', label))
            

def build_file_structure(train_dir, train_ratio):
    # ratio of train/valid dataset
    files = glob.glob(os.path.join(train_dir, '*.jpg'))
    # shuffle the whole training data
    random.shuffle(files)
    
    boundary = (int)(len(files) * train_ratio)

    make_directory(train_dir)
    
    
    # process train dataset
    for file in files[:boundary]:
        filenames = file.split('\\')[-1].split('.')
        os.rename(file, os.path.join(train_dir, 'train', filenames[0], filenames[1]+'.'+filenames[2]))
        
        
    # process valid dataset
    for file in files[boundary:]:
        filenames = file.split('\\')[-1].split('.')
        os.rename(file, os.path.join(train_dir, 'valid', filenames[0], filenames[1]+'.'+filenames[2]))


def dataset_load(dataset_dir, image_transform, batch_size, shuffled, num_workers):
    images = ImageFolder(dataset_dir, image_transform)
    return DataLoader(images,
                      batch_size=batch_size,
                      shuffle=shuffled,
                      num_workers=num_workers)            

