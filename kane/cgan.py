import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between drug sampling")
parser.add_argument("--save_dir", type=str, default="generated/trial_1/", help="directory for saved models, generated examples")
parser.add_argument("--save_interval", type=int, default=100, help="interval between model saving")
opt = parser.parse_args()
print(opt)

def train():
    pass

def sample():
    pass

os.makedirs(opt.save_dir, exist_ok=True)
gen_shape = (1, 51)
cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, gen_shape[-1]),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.model(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + gen_shape[-1], 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, inps, labels):
        # Concatenate label embedding and input
        d_in = torch.cat((inps, self.label_embedding(labels).view(inps.shape[0], opt.n_classes)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader/datasets
class DrugDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.drugs = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        drug = self.drugs.iloc[idx]
        label = np.array([1.], dtype='float64') if str(drug['class']) == 'positive' else np.array([0.], dtype='float64')
        label = torch.from_numpy(label)
        drug = drug.drop(['GeneSym', 'class', 'GeneID'])
        drug = drug.to_numpy(dtype='float64')
        drug = torch.from_numpy(drug)

        if self.transform:
            drug = self.transform(drug)

        return (drug, label)

transform = transforms.Compose([])
dataset = DrugDataset('/Users/kanetian7/omic-features-successful-targets/kane/final_data.csv', transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_out(n_examples, cols, batches_done):
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_examples, opt.latent_dim))))
    labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, n_examples)))
    gen_out = generator(z, labels)

    data = np.concatenate((np.expand_dims(labels.detach().numpy(), axis=1), gen_out.detach().numpy()), axis=1)
    df = pd.DataFrame(data=data, columns=['class'] + cols)
    path = os.path.join(opt.save_dir, 'sampled_' + str(batches_done) + '.csv')
    df.to_csv(path, index=False)

# ----------
#  Training
# ----------

cols = list(pd.read_csv('/Users/kanetian7/omic-features-successful-targets/kane/final_data.csv').columns)[3:]
for epoch in range(opt.n_epochs):
    for i, (inps, labels) in enumerate(dataloader):
        batch_size = inps.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_inps = Variable(inps.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_inps = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_inps, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_inps, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_inps.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_out(n_examples=100, cols=cols, batches_done=batches_done)
    
    if (epoch + 1) % opt.save_interval == 0:
        path = os.path.join(opt.save_dir, 'discriminator_' + str(epoch + 1))
        torch.save(discriminator, path)
        path = os.path.join(opt.save_dir, 'generator_' + str(epoch + 1))
        torch.save(generator, path)
