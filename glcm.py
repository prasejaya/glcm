import skimage
import os, os.path
import csv

from skimage.io import imread
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

file = r'E:\Dataset\hanacaraka32'

im = imread(file)

im = skimage.img_as_ubyte(im)
im //= 64
g = skimage.feature.greycomatrix(im, [1], [0], levels=8, symmetric=False, normed=True)

contrast= skimage.feature.greycoprops(g, 'contrast')[0][0]
energy= skimage.feature.greycoprops(g, 'energy')[0][0]
homogeneity= skimage.feature.greycoprops(g, 'homogeneity')[0][0]
correlation=skimage.feature.greycoprops(g, 'correlation')[0][0]
dissimilarity=skimage.feature.greycoprops(g, 'dissimilarity')[0][0]
ASM=skimage.feature.greycoprops(g, 'ASM')[0][0]


print('contrast is: ', contrast)
print('energy is: ', energy)
print('homogeneity is: ', homogeneity)
print('correlation is: ', correlation)
print('dissimilarity is: ', dissimilarity)
print('ASM is: ', ASM)
	
print("InshaAllah Complete")


   
	
