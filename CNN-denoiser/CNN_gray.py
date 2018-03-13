# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:48:53 2018

@author: xiaoyu.bie
"""

"""
Part 2

In this part, we will establish the CNN with the help of Keras

PS: Pay attention to the noise level in each model
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math 
import numpy as np
import PIL 
import random
import matplotlib.pyplot as plt
# ================================Data Processing=========================================

# some parameters and path
noise = 25 # sigma
folderTest = 'testsets'
CurrentPath = os.getcwd() #'U:/Documents/CNN_denoise'
folderResults = 'results'
PathResults = os.path.join(CurrentPath,folderResults)

# Check the exist of dat folders---X:Noisy Images ; Y: Clean Images
if os.path.isdir(CurrentPath+'/'+folderTest+'/Clean_images_gray'):
    FolderTrainY = folderTest+'/Clean_images_gray'
else: 
    print('Dont have the clean images')
    
if os.path.isdir(CurrentPath+'/'+folderTest+'/Noisy_'+str(noise)+'_images_gray'):
    FolderTrainX = folderTest+'/Noisy_'+str(noise)+'_images_gray'
else: 
    print('Dont have the noisy images')

# create the data path
DataFolderX = os.path.join(CurrentPath,FolderTrainX)
DataFolderY = os.path.join(CurrentPath,FolderTrainY)

# import the data---X:Noisy Images ; Y: Clean Images
(DataX,dimX) = ([],[])
for filename in os.listdir(DataFolderX):
    f_name,f_extension = os.path.splitext(filename)
    if (f_extension.lower() not in ['.tiff']):
        print ("Skipping "+filename+", wrong file type")
        continue
    path = DataFolderX + '/' + filename
    ima = np.asarray(PIL.Image.open(path))
    (nl,nc) = ima.shape
    DataX.append(ima)
    dimX.append((nl,nc))

(DataY,dimY) = ([],[])
for filename in os.listdir(DataFolderY):
    f_name,f_extension = os.path.splitext(filename)
    if (f_extension.lower() not in ['.tiff']):
        print ("Skipping "+filename+", wrong file type")
        continue
    path = DataFolderY + '/' + filename
    ima = np.asarray(PIL.Image.open(path))
    (nl,nc) = ima.shape
    DataY.append(ima)
    dimY.append((nl,nc))

# choose train set, test set and predict set---X:Noisy Images ; Y: Residual Images（=Noisy-Clean）
Index = [i for i in range(len(DataX))]
random.shuffle(Index)
n_train = 6000 # number of train data
n_test = 600 # number of test data
n_predict = 200
seg_size = 32 # size of patchs

TrainX = np.zeros((n_train,seg_size,seg_size))
TrainY = np.zeros((n_train,seg_size,seg_size))
TestX = np.zeros((n_test,seg_size,seg_size))
TestY = np.zeros((n_test,seg_size,seg_size))
for i in range(n_train):
    TrainX[i] = DataX[Index[i]].astype('float32')/255
    TrainY[i] = DataX[Index[i]].astype('float32')/255 - DataY[Index[i]].astype('float32')/255
    
for i in range(n_test):
    TestX[i] = DataX[Index[i+n_train]].astype('float32')/255
    TestY[i] = DataX[Index[i+n_train]].astype('float32')/255 - DataY[Index[i+n_train]].astype('float32')/255
TrainX = np.expand_dims(TrainX,axis = 3)
TrainY = np.expand_dims(TrainY,axis = 3)
TestX = np.expand_dims(TestX,axis = 3)
TestY = np.expand_dims(TestY,axis = 3)



# =================================Build Model===========================================


# import the Keras API
import keras
#from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import Adam
# Build the CNN model
model = Sequential()

# Set the initialization
# init = initializers.lecun_normal(seed = None)
init_k = keras.initializers.RandomNormal(mean = 0, stddev = 0.01, seed = None)
init_b = keras.initializers.Zeros()
# Conv layer 1 
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + ReLU, dilation factor: 1
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 1,
        input_shape = [32,32,1],
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(Activation('relu'))

# Conv layer 2 
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 2
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 2,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))


# Conv layer 3
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 3
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 3,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 4
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 4
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 4,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 5
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 3
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 3,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 6
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 2
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 2,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 7
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution dilation factor: 1
model.add(Conv2D(
        filters = 1,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 1,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))


# Define the optimizer
adam= Adam(lr=1e-3)

# Load the weights
#model.load_weights(PathResults + '/Gray25_random_normal_01.h5')

# Compile the model
model.compile(loss = 'mse', 
              optimizer = adam,
              metrics = ['mse']                     
        )
print(model.summary())

# ====================================Start Train=========================================


epochs = 10
batch_size = 32
model.fit(TrainX,TrainY,
          batch_size = batch_size,
          epochs = epochs,
          validation_data=(TestX, TestY),
          shuffle=True
        )

# ====================================Save weight=========================================
model.save_weights(PathResults + '/Gray'+str(noise)+'_random_normal_01.h5')

# ====================================Check the results=====================================

# preidict the results
PredictX = np.zeros((n_predict,seg_size,seg_size))
PredictY = np.zeros((n_predict,seg_size,seg_size))
CleanX = np.zeros((n_predict,seg_size,seg_size))
for i in range(n_predict):
    PredictX[i] = DataX[Index[i+n_train+n_test]].astype('float32')/255
    CleanX[i] = DataY[Index[i+n_train+n_test]].astype('float32')/255
    
PredictX = np.expand_dims(PredictX,axis = 3)
PredictY = model.predict(PredictX, batch_size=32, verbose=0)

# display the denoising figures
i = 1
CleanImage = CleanX[i]
NoisyImage = np.squeeze(PredictX[i],axis = 2)
ResidualImage = np.squeeze(PredictY[i],axis = 2)
DenoisingImage = NoisyImage - ResidualImage
mse_1 = ((NoisyImage - CleanImage )**2).mean()
PSNR_1 = 20*math.log10(1/np.sqrt(mse_1))
print('The PSNR beteewn noisy image and clean image is: ',PSNR_1)
mse_2 = ((DenoisingImage - CleanImage )**2).mean()
PSNR_2 = 20*math.log10(1/np.sqrt(mse_2))
print('The PSNR beteewn denoising image and clean image is: ',PSNR_2)

plt.figure(num = 'test',figsize = (8,8))
plt.subplot(1,3,1)
plt.title('Clean Image {}'.format(i))
plt.imshow(CleanImage ,cmap = plt.cm.gray)
plt.axis('off')
plt.subplot(1,3,2)
plt.title('Noisy image {}'.format(i))
plt.imshow(NoisyImage ,cmap = plt.cm.gray)
plt.axis('off')
plt.subplot(1,3,3)
plt.title('Denoising Image + {}'.format(i))
plt.imshow(DenoisingImage,cmap = plt.cm.gray)
plt.axis('off')


# ===============================Implementation in practical images================

noise = 25
# load the practical images
imageSets = ['BSD68','Set12']
folderTest = 'testsets'
ImageFolder = os.path.join(CurrentPath,folderTest,imageSets[0])
(images,dim,id) = ([],[],0)
id = 0 # lable of images
for filename in os.listdir(ImageFolder):
    f_name, f_extension = os.path.splitext(filename)
    if (f_extension.lower() not in
            ['.png','.jpg','.jpeg','.gif','.pgm']):
        print("Skipping "+filename+", wrong file type")
        continue
    path = ImageFolder + '/' + filename
    ima = np.asarray(PIL.Image.open(path))
    (nl,nc) = ima.shape
    images.append(ima.flatten())
    dim.append((nl,nc))

# choose an image
i = 1
if dim[i] == (481,321):
    Original_image = images[i].reshape(dim[i]).astype(float)
elif dim[i] == (321,481):
    Original_image = images[i].reshape(dim[i]).astype(float).T
else:
    print('Be aware of the size of input image')
Input_shape = list(dim[i])
Input_shape.append(1) 
nl, nc = dim[i]

Noisy_image = Original_image + noise*np.random.randn(nl,nc)
Noisy_image[Noisy_image<0] = 0
Noisy_image[Noisy_image>255] = 255

Original_image = Original_image/255
Noisy_image = Noisy_image/255

mse_1 = ((Noisy_image  - Original_image )**2).mean()
PSNR_1 = 20*math.log10(1/np.sqrt(mse_1))

#                             Build a new model
# import the Keras API
import keras
#from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import Adam
# Build the CNN model
model = Sequential()

# Set the initialization
# init = initializers.lecun_normal(seed = None)
init_k = keras.initializers.RandomNormal(mean = 0, stddev = 0.01, seed = None)
init_b = keras.initializers.Zeros()


# Conv layer 1 
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + ReLU, dilation factor: 1
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 1,
        input_shape = (481,321,1),
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(Activation('relu'))

# Conv layer 2 
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 2
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 2,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))


# Conv layer 3
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 3
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 3,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 4
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 4
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 4,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 5
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 3
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 3,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 6
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution + Batch Normalization + ReLU, dilation factor: 2
model.add(Conv2D(
        filters = 64,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 2,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Conv layer 7
## input shape (32,32,1)
## output shape (32,32,32)
## Dilated Convolution dilation factor: 1
model.add(Conv2D(
        filters = 1,
        kernel_size = (3,3),
        padding = 'same',
        dilation_rate = 1,
        kernel_initializer= init_k,
        bias_initializer= init_b
        ))


# Define the optimizer
adam= Adam(lr=1e-3)

# Load the weights
model.load_weights(PathResults + '/Gray'+str(noise)+'_random_normal_01.h5')

# Compile the model
model.compile(loss = 'mse', 
              optimizer = adam,
              metrics = ['mse']                     
        )
print(model.summary())


Image_in = np.expand_dims(Noisy_image,axis = 0)
Image_in = np.expand_dims(Image_in,axis = 3)
Image_out = model.predict(Image_in, batch_size=32, verbose=0)
Residual_image = np.squeeze(Image_out[0],axis = 2)
Denoise_image = Noisy_image - Residual_image
mse_2 = ((Denoise_image  - Original_image )**2).mean()
PSNR_2 = 20*math.log10(1/np.sqrt(mse_2))
print('The PSNR beteewn noisy image and clean image is: ',PSNR_1)
print('The PSNR beteewn denoising image and clean image is: ',PSNR_2)

plt.figure(num = 'test',figsize = (8,8))
plt.subplot(1,3,1)
plt.title('Clean Image {}'.format(i))
plt.imshow(Original_image ,cmap = plt.cm.gray)
plt.axis('off')
plt.subplot(1,3,2)
plt.title('Noisy image {}'.format(i))
plt.imshow(Noisy_image ,cmap = plt.cm.gray)
plt.axis('off')
plt.subplot(1,3,3)
plt.title('Denoising Image + {}'.format(i))
plt.imshow(Denoise_image,cmap = plt.cm.gray)
plt.axis('off')

