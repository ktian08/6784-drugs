import os
import numpy as np
import math
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(2, 2)
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + 2, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 51),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.model(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(2, 2)

        self.model = nn.Sequential(
            nn.Linear(2 + 51, 512),
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
        d_in = torch.cat((inps, self.label_embedding(labels).view(inps.shape[0], 2)), -1)
        validity = self.model(d_in)
        return validity

class DrugDataset(Dataset):
    def __init__(self, X_tr, Y_tr, transform=None):
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.transform = transform

    def __len__(self):
        return self.X_tr.shape[0]

    def __getitem__(self, idx):
        drug = self.X_tr[idx, :]
        drug = torch.from_numpy(drug)
        label = self.Y_tr[idx]
        label = torch.from_numpy(label)

        if self.transform:
            drug = self.transform(drug)

        return (drug, label)

# def setup():
#     lr = 0.0002
#     b1 = 0.5
#     b2 = 0.999
#     latent_dim = 100
#     adversarial_loss = torch.nn.MSELoss()
#     transform = transforms.Compose([])

#     n_epochs = 200
#     batch_size = 64
#     n_cpu = 8
#     n_classes = 2
#     gen_shape = (1,51)
#     sample_interval = 400
#     save_dir = 'generated/trial_1/'
#     save_interval = 100
    
#     dataset = DrugDataset('/Users/kanetian7/omic-features-successful-targets/kane/final_data.csv', transform=transform)
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#     )

#     return (n_epochs, lr, b1, b2, latent_dim, n_cpu, n_classes, 
#         sample_interval, save_dir, save_interval, gen_shape, adversarial_loss)

def train(X_tr, Y_tr, params):
    (n_epochs, batch_size, transform, lr, b1, b2, wd, latent_dim, adversarial_loss) = params

    dataset = DrugDataset('/Users/kanetian7/omic-features-successful-targets/kane/final_data.csv', transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)

    for epoch in range(n_epochs):
        for i, (inps, labels) in enumerate(dataloader):
            batch_size = inps.shape[0]

            valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            real_inps = Variable(inps.type(torch.FloatTensor))
            labels = Variable(labels.type(torch.LongTensor))

            optimizer_G.zero_grad()

            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(torch.LongTensor(np.random.randint(0, 2, batch_size)))

            gen_inps = generator(z, gen_labels)

            validity = discriminator(gen_inps, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            validity_real = discriminator(real_inps, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            validity_fake = discriminator(gen_inps.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
    return generator, discriminator


def sample_out(generator, n_examples, cl):
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (generator.n_examples, generator.latent_dim))))
    labels = Variable(torch.LongTensor(np.full(generator.n_examples, cl)))
    gen_out = generator(z, labels)

    data = gen_out.detach().numpy()
    return data