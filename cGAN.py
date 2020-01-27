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

#reshape layer
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)



# Generator Class
class Generator(nn.Module): 
	def __init__(self, n_classes,latentdim, batch_size, dataset_name, img_shape):
		super(Generator, self).__init__()
		self.label_embed = nn.Embedding(n_classes, n_classes) # mi crea un dizionario di 10 elementi, ogni elemento è a sua volta un vettore di 10 elementi
		self.dataset_name=dataset_name
		self.img_shape=img_shape
		self.depth = 8000 # dimensione output primo layer

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.BatchNorm1d(output, 0.8)) # do also batch normalization after a layer, then apply leakyRelu as activation function
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.generator = nn.Sequential(
			*init(latentdim + n_classes, self.depth*batch_size),
			#*init(self.depth, self.depth * 2), 
			#*init(self.depth * 2, self.depth * 4), 
			#*init(self.depth * 4, self.depth * 8),
            nn.Linear(self.depth*batch_size, self.depth*batch_size),
            nn.Sigmoid(),
			Reshape(batch_size, 80, 10, 10),
            nn.ConvTranspose2d(80, 3, 5, 3, bias=False)
            # nn.Linear(self.depth * 8, int(np.prod(img_shape))), # np.prod ritorna il prodotto dei valori sugli axes - in questo caso il prodotto delle dimensioni dell'immagine
            # nn.Tanh()    
			)

	# torchcat needs to combine tensors --> l'embedding delle features sta tutto qui...
	def forward(self, noise, labels):
		if self.dataset_name!='celeb':
			print("Requested labels", labels.size(), labels)
			# in pratica ogni label che noi vogliamo (es: digit 9, digit 3..) fa da chiave nel dizionario label_embed (una hash table) a un vettore di 10 elementi. Questi 10 elementi sono casuali e diversi per ogni label (es: la label 3 sarà una roba tipo [-0.24, 0-7...] con 10 elementi)
			# label_embed(labels) avrà quindi 64 (dimensione di un batch che produciamo alla volta, ergo 64 immagini finte) x10 (ogni label richiesta come detto è tradotta in un vettore di 10 elementi)
			# il noise ha dimensione 64x100 (64 immagini, un immagine per ogni riga )
			# ECCO IL MISTERIOSO EMBEDDING
			# al noise vengono aggiunte 10 colonne, che sono le 10 colonne delle label, quindi ogni riga di gen_input è 100 pixel di noise + 10 float che rappresentano la label che vogliamo per quell'immagine. Questo è l'embedding
			gen_input = torch.cat((self.label_embed(labels), noise), -1)    
			print("conditional vector size", self.label_embed(labels).size())
			print("Input to generator size",gen_input.size())
		else:
			'''
				tentativo1: concateniamo al noise direttamente il vettore binario coi 40 attributi, senza nessun mapping. labels sarà una matrice binaria
				di 64x40
			'''
		#	print("Requested labels", labels.size(), labels)
			gen_input = torch.cat((labels, noise), -1)    
		#	print("conditional vector size", labels.size())
		#	print("Input to generator size",gen_input.size())
			
		img = self.generator(gen_input)
		img = img.view(img.size(0), *self.img_shape) # view è un reshape per ottenere dal vettore in output un immagine con le 64 immagini generate dentro
		return img

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m

class MaxoutConv(nn.Module):
    def __init__(self,pool_size, kernel_size, stride):
        super(MaxoutConv, self).__init__()

        self.discriminator = nn.Sequential(
            Maxout(pool_size),
            nn.MaxPool2d()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x


def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
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


def forward(self, input):
    return self.main(input)