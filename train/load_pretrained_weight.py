import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu")

# Hyper-parameters
latent_size = 5184
hidden_size_G = 1728
hidden_size_D = 108
image_size = 216
readout_lines = 384
num_epochs = 10
input_size = 24
batch_size = 5
test_total_slices = 13824
sample_dir = '/home/nhjeong/db'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# Discriminator
D = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 8, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 2, 3, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(image_size * readout_lines * 2, image_size),
    nn.ReLU(),
    nn.Linear(image_size, 1),
    nn.Sigmoid())

# Generator
G = nn.Sequential(
    nn.Conv2d(input_size, hidden_size_G, kernel_size=(image_size, 1)),
    nn.ReLU(),
    nn.Conv2d(hidden_size_G, hidden_size_G, 1),
    nn.ReLU(),
    nn.Conv2d(hidden_size_G, hidden_size_G, 1),
    nn.ReLU(),
    nn.Conv2d(hidden_size_G, image_size, 1))


G = torch.load('weight1.pkl')

# Device setting
D = D.to(device)
G = G.to(device)

# MSE, Binary cross entropy loss and optimizer
baseloss = nn.MSELoss()
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-5)
g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, train=True):

        self.train = train

        if self.train:
            self.train_X_mat = h5py.File('/home/nhjeong/db/db.mat', 'r')
            self.train_X_input = self.train_X_mat['db'][:]
            self.train_Y_mat = h5py.File('/home/nhjeong/db/gt.mat', 'r')
            self.train_Y_input = self.train_Y_mat['gt'][:]
            self.train_X_mat.close()
            self.train_Y_mat.close()

        else:
            self.test_X_mat = h5py.File('/home/nhjeong/db/test_db.mat', 'r')
            self.test_X_input = self.test_X_mat['test_db'][:]

            self.test_Y_mat = h5py.File('/home/nhjeong/b/test_gt.mat', 'r')
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

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)


# Start training
total_step = len(trainloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.permute(0, 3, 1, 2)
        labels = labels.view((batch_size, 1, 216, 384))
        index = np.random.randint(-6, 7)
        images = np.roll(images, index, axis=2)
        labels = np.roll(labels, index, axis=2)
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(labels)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        fake_images = G(images)
        fake_images = fake_images.permute(0, 2, 1, 3)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        fake_images = G(images)
        fake_images = fake_images.permute(0, 2, 1, 3)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        base_loss = baseloss(fake_images, labels)
        generator_loss = criterion(outputs, real_labels)
        g_loss = 0.5 * generator_loss + 0.5 * base_loss

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 10 == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}, MSE: {:.4f}'
                .format(epoch + 1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                        real_score.mean().item(), fake_score.mean().item(), base_loss.item()))
