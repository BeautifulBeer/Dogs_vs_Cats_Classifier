#!/usr/bin/env python
# coding: utf-8

# In[8]:


from torchvision.models import resnet50
from torch.autograd import Variable
import torch
import time
import importlib
import preprocessing


# In[9]:


'''
Transfer learning is prevalent methodology to learn DNN(deep neural network) using pretrained DNN.
In this method, resnet50 is used to build a classifier of dogs and cats.
'''
def get_model():
    # cuda release cached memory
    torch.cuda.empty_cache()
    model_ft = resnet50(pretrained=True)
    input_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(input_features, 2)
    if torch.cuda.is_available():
        model_ft = model_ft.cuda()
    return model_ft

'''
Adam optimizer, StepLR scheduler and crossEntropyLoss are used.
'''
def training(model_ft, datasets, epochs=25, learning_rate=0.001):
    begin_time = time.time()
    
    datasets_size = {'train': len(datasets['train'].dataset), 'valid' : len(datasets['valid'].dataset)}
        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_corrects = 0.0
    
    for epoch in range(epochs):
        epoch_time = time.time()
        print(f'----------------- Epoch {epoch} -----------------')
        for mode in ['train', 'valid']:
            if mode == 'train':
                model_ft.train(True)
            else:
                model_ft.train(False)
            
            total_corrects = 0
            total_loss = 0
                        
            # mini batch
            for data in datasets[mode]:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                if mode == 'train':
                    optimizer.zero_grad()

                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item()
                total_corrects += torch.sum(preds == labels).item()
            
            if mode == 'valid':
                if best_corrects < total_corrects:
                    best_corrects = total_corrects
                    torch.save(model_ft.state_dict(), 'dogs_vs_cats_classifier_res50.pth')
            
            print(f'{mode} Error : {(total_loss / datasets_size[mode]):.5f}')
            print(f'{mode} Correct : {(total_corrects / datasets_size[mode]):.5f}')
        epoch_end_time = time.time()
        print(f'Time : {int((epoch_end_time - epoch_time) / 60)}M {int((epoch_end_time - epoch_time) % 60)}s')
        # decay learning rate according to epoch
        lr_scheduler.step()
    end_time = time.time()
    print(f'Total Training Time : {int((end_time - begin_time) / 60)}M {int((end_time - begin_time) % 60)}s')

