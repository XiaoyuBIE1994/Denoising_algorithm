# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:07:32 2018
This is the reproduce for article 'Learning Deep CNN Denoiser Prior for Image Restoration'
In this part, we reproduce the denoising algotithm in gray image
@author: xiaoyu.bie
"""

"""
Part 1 

In this part, we will define the all the necessary parameters
and import all the image data and segment the images if needed
"""

import sys
import os

import numpy as np
import PIL 


print(os.getcwd())
#　define the name of folder
imageSets = ['BSD68','Set12']
folderTest = 'testsets'
folderModel = 'models'
folderResult = 'results'
taskTestCur = 'Denoising'
CurrentPath =  os.getcwd()#'U:/Documents/CNN_denoise'

FunctionPath = CurrentPath+'/utilities'
sys.path.append(FunctionPath)

# create folder to store the results
if not os.path.isdir(CurrentPath+'/'+folderResult):
    os.makedirs(CurrentPath+'/'+folderResult)


# import the image data
ImageFolder = os.path.join(CurrentPath,folderTest,imageSets[0])
ResultFolder = os.path.join(CurrentPath,folderResult,imageSets[0])

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

# segment the images into small size(limit of computing power): chosse 100 
# patches per image 
SegmentationFolder = 'BSD68_seg'
seg_size = 32
images_seg = [None] * (len(images)*100)
j = 0
for i in range(len(images)):
    im_tem = images[i].reshape(dim[i])
    for m in range(10):
        for n in range(10):
            images_seg[j] = im_tem[seg_size*m:seg_size*(m+1),seg_size*n:seg_size*(n+1)]
            j += 1

# save clean images
if not os.path.isdir(CurrentPath+'/'+folderTest+'/Clean_images_gray'):
    os.makedirs(CurrentPath+'/'+folderTest+'/Clean_images_gray')
for i in range(len(images_seg)):
    a = PIL.Image.fromarray(images_seg[i].astype(float))
    a.save(CurrentPath+'/'+folderTest+'/Clean_images_gray/'+'{}'.format(i)+'.tiff')

# add noise
noise = 35
images_noisy = [None] * (len(images)*100)
for i in range(len(images_seg)):
    images_noisy[i] = images_seg[i].astype(float)+noise*np.random.randn(seg_size,seg_size)
    images_noisy[i][images_noisy[i]<0] = 0
    images_noisy[i][images_noisy[i]>255] = 255
    
# save noisy images
if not os.path.isdir(CurrentPath+'/'+folderTest+'/Noisy_'+str(noise)+'_images_gray'):
    os.makedirs(CurrentPath+'/'+folderTest+'/Noisy_'+str(noise)+'_images_gray')
for i in range(len(images_noisy)):
    a = PIL.Image.fromarray(images_noisy[i])
    a.save(CurrentPath+'/'+folderTest+'/Noisy_'+str(noise)+'_images_gray/'+'{}'.format(i)+'.tiff')



'''
# test the data
plt.figure(num = 'test',figsize = (8,8))
i = 1
plt.subplot(2,2,1)
plt.title('Image {}'.format(i))
plt.imshow(images_seg[i],cmap = plt.cm.gray)
plt.axis('off')
i = 10
plt.subplot(2,2,2)
plt.title('Image + {}'.format(i))
plt.imshow(images_seg[i],cmap = plt.cm.gray)
plt.axis('off')
i = 21
plt.subplot(2,2,3)
plt.title('Image + {}'.format(i))
plt.imshow(images_seg[i],cmap = plt.cm.gray)
plt.axis('off')
i = 36
plt.subplot(2,2,4)
plt.title('Image + {}'.format(i))
plt.imshow(images_seg[i],cmap = plt.cm.gray)
plt.axis('off')
'''





        



    
    

