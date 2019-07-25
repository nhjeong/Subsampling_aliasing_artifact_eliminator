import torch
import torch.nn as nn
import torch.utils.data
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


# Hyper-parameters
latent_size = 5184
hidden_size = 1728
image_size = 216
num_epochs = 200
batch_size = 128
sample_dir = '/home/nhjeong/MLPGAN/db'    # Directory of database


# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size))


# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")


G = torch.load('weight_sample.pkl')



class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, train=True):
        
        self.train = train
        
        if self.train:
            self.train_X_mat = h5py.File(os.path.join(sample_dir, 'db.mat'), 'r')
            self.train_X_input = self.train_X_mat['db'][:]


            self.train_Y_mat = h5py.File(os.path.join(sample_dir, 'gt.mat'), 'r')
            self.train_Y_input = self.train_Y_mat['gt'][:]

            self.train_X_mat.close()
            self.train_Y_mat.close()

        else:
            self.test_X_mat = h5py.File(os.path.join(sample_dir, 'test_db.mat'), 'r')
            self.test_X_input = self.test_X_mat['test_db'][:]          

            self.test_Y_mat = h5py.File(os.path.join(sample_dir, 'test_gt.mat'), 'r')
            self.test_Y_input = self.test_Y_mat['test_gt'][:]           

            self.test_X_mat.close()
            self.test_Y_mat.close()
        
        
        
    def __len__(self):
        if self.train:
            return self.train_X_input.shape[0]
        else:
            return self.test_X_input.shape[0]
    
    def __getitem__(self, index):
        if self.train:
            raw, target = self.train_X_input[index,], self.train_Y_input[index,]
        else:
            raw, target = self.test_X_input[index,], self.test_Y_input[index,]
            
        return raw, target
    
    
testset = MyDataset(train=False)
output = G(torch.tensor(testset.test_X_input).to(device))
test_result = output.cpu().detach().numpy()



nrmse = []
for i in range(36):
    tmp = testset.test_X_input[384*i:384*(i+1),0:2592] + 1j*testset.test_X_input[384*i:384*(i+1),2592:5184]
    undersampled = np.zeros((384, 216))
    for k in range(12):
        undersampled += np.abs(tmp[:,k*216:(k+1)*216])
    ans = testset.test_Y_input[384*i:384*(i+1),:]
    pred = test_result[384*i:384*(i+1),:]
    error = ans - pred
    rmse = (np.sum(error ** 2) / np.sum(ans ** 2)) ** 0.5
    plt.figure(figsize=[10, 40])
    plt.subplot(1,4,1)
    plt.imshow(undersampled)
    plt.axis('off')      
    plt.subplot(1,4,2)
    plt.imshow(pred)
    plt.axis('off')      
    plt.subplot(1,4,2)
    plt.imshow(pred)
    plt.axis('off')       
    plt.subplot(1,4,3)
    plt.imshow(ans)
    plt.axis('off') 
    plt.subplot(1,4,4)
    plt.imshow(np.abs(error)*2, clim=[0,1])
    plt.axis('off') 
    plt.show()    
    nrmse.append(rmse)
print('nRMSE: %.3lf %%' % (np.mean(nrmse)*100))
