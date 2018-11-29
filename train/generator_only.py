import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
# % matplotlib inline           # When drawing a plot for something at iPython environment

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")          # Assigning 0~3 (4 GPUs for 1 workstation)

# Hyper-parameters
latent_size = 5184
hidden_size_G = 1728
hidden_size_D = 108
image_size = 216
in_channel = 24             # 12-channel and real-imaginary
num_epochs = 500
batch_size = 5
test_total_slices = 13824
lr = 1e-4
sample_dir = '/home/nhjeong/MLPGAN/db'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# Generator
G = nn.Sequential(
    nn.Conv2d(in_channel, hidden_size_G, kernel_size=(image_size, 1)),    # in_channels, out_channels, kernel_size
    nn.ReLU(),
    nn.Conv2d(hidden_size_G, hidden_size_G, 1),                 # 1x1 convolution
    nn.ReLU(),
    nn.Conv2d(hidden_size_G, hidden_size_G, 1),
    nn.ReLU(),
    nn.Conv2d(hidden_size_G, image_size, 1))

# Device setting
G = G.to(device)

# MSE, Binary cross entropy loss and optimizer
baseloss = nn.MSELoss()
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)


def reset_grad():
    g_optimizer.zero_grad()




class MyDataset(torch.utils.data.Dataset):

    def __init__(self, train=True):

        self.train = train

        if self.train:
            self.train_X_mat = h5py.File('/home/nhjeong/MLPGAN/db/db_gan2.mat', 'r')
            self.train_X_input = self.train_X_mat['db'][:]

            self.train_Y_mat = h5py.File('/home/nhjeong/MLPGAN/db/gt_gan2.mat', 'r')
            self.train_Y_input = self.train_Y_mat['gt'][:]

            self.train_X_mat.close()
            self.train_Y_mat.close()

        else:
            self.test_X_mat = h5py.File('/home/nhjeong/MLPGAN/db/test_db_gan2.mat', 'r')
            self.test_X_input = self.test_X_mat['test_db'][:]

            self.test_Y_mat = h5py.File('/home/nhjeong/MLPGAN/db/test_gt_gan2.mat', 'r')
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


trainset = MyDataset(train=True)
testset = MyDataset(train=False)

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)


train_loss = []

# Start training
total_step = len(trainloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.permute(0, 3, 1, 2)
        labels = labels.view((batch_size, 1, 216, 384))
        index = np.random.randint(-6, 7)                    # steps of shift augmentation (Maximum 6 pixels)
        images = np.roll(images, index, axis=2)
        labels = np.roll(labels, index, axis=2)
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)


        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        fake_images = G(images)
        labels = labels.permute(0, 2, 1, 3)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        base_loss = baseloss(fake_images, labels)
        train_loss.append(base_loss.data[0])

        # Backprop and optimize
        reset_grad()
        base_loss.backward()
        g_optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], MSE: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, base_loss.item()))
