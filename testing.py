#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
from PIL import Image
import pandas as pd
import glob
import torch


# In[4]:


'''
Define dataset class for test dataset
'''
class test_dataset(Dataset):
    def __init__(self, path, transform):
        self.files = glob.glob(path)
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.files[idx]))
        label = self.files[idx].split('\\')[-1].split('.')[0]
        return img, label

def testing(model, test_data):
    result = []
    test_data_gen = DataLoader(test_data, batch_size=1)
    for data in test_data_gen:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        result.append([labels[0], pred.item()])
    return result

'''
It's necessary to export predicted result as a csv file to participate kaggle competition, "dogs vs cats redux kernel edition".
'''
def export_submission(result):
    result.sort(key=lambda element: int(element[0]))
    pd.DataFrame(result, columns=['id', 'label']).to_csv('submission.csv', index=False)

