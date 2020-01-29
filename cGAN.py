import numpy as np
import os
import random

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
from torchvision.datasets.utils import download_file_from_google_drive

from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


# reshape layer
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


preferred_dataset = 'lfwcrop'


# Generator Class
class Generator(nn.Module):
    def __init__(self, n_classes, latentdim, batch_size, dataset_name, img_shape):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(n_classes,
                                        n_classes)  # mi crea un dizionario di 10 elementi, ogni elemento è a sua volta un vettore di 10 elementi
        self.dataset_name = dataset_name
        self.img_shape = img_shape
        self.depth = 8000  # dimensione output primo layer

        def init(input, output, normalize=True):
            layers = [nn.Linear(input, output)]
            if normalize:
                layers.append(nn.BatchNorm1d(output,
                                             0.8))  # do also batch normalization after a layer, then apply leakyRelu as activation function
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generator_step1 = nn.Sequential(
            nn.Linear(latentdim + n_classes, self.depth),
            nn.Sigmoid()
        )
        self.generator_step2=nn.Sequential(
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.ConvTranspose2d(80, 40, 3, 1, bias=False),  # out 12
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.ConvTranspose2d(40, 20, 7, 1, bias=False),  # out 18
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 3, 13, 3, bias=False),  # out 64
            nn.ReLU(),
            #PrintLayer("fineCreazione"),
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), #64x32x32
            nn.LeakyReLU(0.2, inplace=True),
            #PrintLayer("dopo Conv1"),
            nn.MaxPool2d(6, stride=1, padding=2, dilation=4), #64x16x16
            #PrintLayer("dopoMaxPool1"),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),#64*2x8x8
            #PrintLayer("Conv2"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4, stride=2, padding=1, dilation=1), #64*2x4x4
            #PrintLayer("dopoMaxPool2")
        )
        self.generator_step3 = nn.Sequential(
            nn.Linear(64*2*4*4,3*64*64),
            nn.LeakyReLU(0.2, inplace=True)
        )

    # torchcat needs to combine tensors --> l'embedding delle features sta tutto qui...
    def forward(self, noise, labels, b_size):
        if self.dataset_name != preferred_dataset:
            print("Requested labels", labels.size(), labels)
            # in pratica ogni label che noi vogliamo (es: digit 9, digit 3..) fa da chiave nel dizionario label_embed (una hash table) a un vettore di 10 elementi. Questi 10 elementi sono casuali e diversi per ogni label (es: la label 3 sarà una roba tipo [-0.24, 0-7...] con 10 elementi)
            # label_embed(labels) avrà quindi 64 (dimensione di un batch che produciamo alla volta, ergo 64 immagini finte) x10 (ogni label richiesta come detto è tradotta in un vettore di 10 elementi)
            # il noise ha dimensione 64x100 (64 immagini, un immagine per ogni riga )
            # ECCO IL MISTERIOSO EMBEDDING
            # al noise vengono aggiunte 10 colonne, che sono le 10 colonne delle label, quindi ogni riga di gen_input è 100 pixel di noise + 10 float che rappresentano la label che vogliamo per quell'immagine. Questo è l'embedding
            gen_input = torch.cat((self.label_embed(labels), noise), -1)
            print("conditional vector size", self.label_embed(labels).size())
            print("Input to generator size", gen_input.size())
        else:
            '''
                tentativo1: concateniamo al noise direttamente il vettore binario coi 40 attributi, senza nessun mapping. labels sarà una matrice binaria
                di 64x40
            '''
            #	print("Requested labels", labels.size(), labels)
            gen_input = torch.cat((labels, noise), -1)
        #	print("conditional vector size", labels.size())
        #	print("Input to generator size",gen_input.size())

        step1 = self.generator_step1(gen_input)
        reshape=step1.view(b_size,80,10,10)
        img=self.generator_step2(reshape)
        reshape2=img.view(b_size,64*2*4*4,1)
        img=self.generator_step3(reshape2).view(b_size,3, 64, 64)
        img = img.view(img.size(0),
                       *self.img_shape)  # view è un reshape per ottenere dal vettore in output un immagine con le 64 immagini generate dentro
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, latentdim, batch_size, img_shape, dataset_name, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.depth = 64 * 64 * nc
        self.dataset_name = dataset_name
        self.linear = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), self.depth),
            nn.LeakyReLU(),
        )
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels, b_size):
        imgs = img.view(img.size(0), -1)
        if self.dataset_name == preferred_dataset:
            inpu = torch.cat((imgs, labels.float()), -1)
        else:
            inpu = torch.cat((imgs, self.label_embed1(labels)),
                             -1)  # associa all'immagine generata (che contiene più cifre da riconoscere) le labels che erano state richieste

        inpu = self.linear(inpu).view(b_size, 3, 64, 64)
        validity = self.main(inpu)
        return validity

class PrintLayer(nn.Module):
    def __init__(self, layer):
        super(PrintLayer, self).__init__()
        self.layer=layer

    def forward(self, x):
        # Do your print / debug stuff here
        print("Shape after %s is "%self.layer, x.size())
        return x