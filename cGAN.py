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

# Generator Class
class Generator(nn.Module): 
	def __init__(self):
		super(Generator, self).__init__()
		self.label_embed = nn.Embedding(n_classes, n_classes) # mi crea un dizionario di 10 elementi, ogni elemento è a sua volta un vettore di 10 elementi
		self.depth = 128 # dimensione output primo layer

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.BatchNorm1d(output, 0.8)) # do also batch normalization after a layer, then apply leakyRelu as activation function
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.generator = nn.Sequential(
			*init(latentdim + n_classes, self.depth), 
			*init(self.depth, self.depth * 2), 
			*init(self.depth * 2, self.depth * 4), 
			*init(self.depth * 4, self.depth * 8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))), # np.prod ritorna il prodotto dei valori sugli axes - in questo caso il prodotto delle dimensioni dell'immagine
            nn.Tanh()    
			)

	# torchcat needs to combine tensors --> l'embedding delle features sta tutto qui...
	def forward(self, noise, labels): 
		if dataset_name!='celeb':
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
		img = img.view(img.size(0), *img_shape) # view è un reshape per ottenere dal vettore in output un immagine con le 64 immagini generate dentro
		return img

class Discriminator(nn.Module): 
	def __init__(self): 
		super(Discriminator, self).__init__()
		self.label_embed1 = nn.Embedding(n_classes, n_classes)
		self.dropout = 0.4 
		self.depth = 512

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.Dropout(self.dropout))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.discriminator = nn.Sequential(
			*init(n_classes + int(np.prod(img_shape)), self.depth, normalize=False),
			*init(self.depth, self.depth), 
			*init(self.depth, self.depth),
			nn.Linear(self.depth, 1),
			nn.Sigmoid()  # classify as true or false
			)

	def forward(self, img, labels): 
		imgs = img.view(img.size(0), -1)
		if dataset_name=='celeb':  
			inpu = torch.cat((imgs, labels.float()), -1)
		else:	
			inpu = torch.cat((imgs, self.label_embed1(labels)), -1) # associa all'immagine generata (che contiene più cifre da riconoscere) le labels che erano state richieste
		
		validity = self.discriminator(inpu)
		return validity 