#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 06:36:55 2018

@author: sunitapattanayak
"""

import numpy
import numpy.random
import os
from os import listdir
from os.path import isfile, join,basename
#Importing the openCV python library for image processing
import cv2
import warnings
warnings.filterwarnings("ignore")

# Importing the Keras libraries and packages
from keras.models import Sequential, Model
from keras.layers import Dense,Flatten, Activation, BatchNormalization, Conv2D, Lambda, Input
from keras.layers.convolutional import MaxPooling2D
from keras.layers import merge
from keras.optimizers import Adam

#initialise image array
t_x=[]
t_y=[]
WIDTH = 128
HEIGHT = 128


#Read all images using cv2 package
AND_Dataset_path='./AND_c'
onlyfiles = [ f for f in listdir(AND_Dataset_path) if isfile(join(AND_Dataset_path,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for x in range(0, len(onlyfiles)):
    image_path = os.path.join(AND_Dataset_path, onlyfiles[x])
    if os.path.exists(image_path):
        images[x] = cv2.imread( image_path )
        t_x.append(cv2.resize(images[x], (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        t_y.append(numpy.int64(basename(onlyfiles[x])[:4]))

t_x=numpy.asarray(t_x)
t_y=numpy.asarray(t_y)

#Split dataset into train and test in ratio 4:1
from sklearn.model_selection import train_test_split
train_X , test_X ,train_Y ,test_Y= train_test_split(t_x,t_y,train_size=0.80,test_size=0.20,shuffle=False)

print(train_X.shape,test_X.shape)

#Generating CNN model
input_shape = (128,128,3)

img_imput1 = Input(input_shape)
img_imput2 = Input(input_shape)

model = Sequential()

model.add(Conv2D(8, (3,3), strides=(1, 1), input_shape = (128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(16, (3,3), strides=(1, 1), input_shape = (128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3,3), strides=(1, 1), input_shape = (128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3,3), strides=(1, 1), input_shape = (128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3,3), strides=(1, 1), input_shape = (128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3,3), strides=(1, 1), input_shape = (128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

model.add(Dense(512,kernel_initializer='random_uniform', name='dense1'))  
model.add(BatchNormalization())
model.add(Activation('relu'))
# 
model.add(Dense(1024,name='output'))
model.add(Activation('sigmoid'))  

#encode into the model
encoded_left = model(img_imput1)
encoded_right = model(img_imput2)     

#merge the inputs using L1 distance between them
#Chi Square distance calculation and merge
chisqdist = Lambda(lambda x: (x[0]-x[1])**2/(x[0]+x[1]))
both_imgs=merge([encoded_left,encoded_right],mode = chisqdist, output_shape=lambda x: x[0])

prediction = Dense(2,activation='sigmoid',name='prediction')(both_imgs)

siamese = Model(inputs=[img_imput1,img_imput2],outputs=prediction)

optimizer = Adam(0.00006)
#Loss function used : Binary cross entropy
siamese.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

siamese.count_params()

#Generating Same class data and Different class data
from itertools import zip_longest

train_y=[]
train_label=[] 
train_img_1=[]
train_img_2=[]
count1=0
count2=0

for (img1,lab1) in zip_longest(train_X, train_Y):
    count3=0
    count4=0
    for (img2,lab2) in zip_longest(train_X, train_Y):
        if(count3>5 or count4>5):
            break
        if(not numpy.array_equal(img1,img2)):
            if(lab1==lab2):
                train_y.append(lab1)
                train_label.append([1,0])
                count1+=1
                count3+=1
            else:
                if(count2>count1):
                    continue
                train_y.append(str(lab1)+" "+str(lab2))
                train_label.append([0,1])
                count2+=1
                count4+=1
            train_img_1.append(numpy.asarray(img1))
            train_img_2.append(numpy.asarray(img2))
        
siamese.fit(x=[numpy.asarray(train_img_1),numpy.asarray(train_img_2)], y=numpy.asarray(train_label), batch_size=50, epochs=3, verbose=1)
#Saving weights
fname="weights-project-CNN-2.hdf5"
siamese.save_weights(fname,overwrite=True)
#Loading back weights
fname="weights-project-CNN-2.hdf5"
siamese.load_weights(fname)
#loss = siamese.train_on_batch([numpy.asarray(train_img_1),numpy.asarray(train_img_2)],numpy.asarray(train_label))

train_eval = siamese.evaluate(x=[numpy.asarray(train_img_1),numpy.asarray(train_img_2)],y=numpy.asarray(train_label),verbose=1)
print('Train loss:', train_eval[0])
print('Train accuracy:', train_eval[1])

#Test data
test_y=[]
test_label=[] 
test_img_1=[]
test_img_2=[]
count1=0
count2=0

for (img1,lab1) in zip_longest(test_X, test_Y):
    count3=0
    count4=0
    for (img2,lab2) in zip_longest(test_X, test_Y):
        if(count3>5 or count4>5):
            break
        if(not numpy.array_equal(img1,img2)):
            if(lab1==lab2):
                test_y.append(lab1)
                test_label.append([1,0])
                count1+=1
                count3+=1
            else:
                if(count2>count1):
                    continue
                test_y.append(str(lab1)+" "+str(lab2))
                test_label.append([0,1])
                count2+=1
                count4+=1
            test_img_1.append(numpy.asarray(img1))
            test_img_2.append(numpy.asarray(img2))
#Finding probability
prob = siamese.predict([numpy.asarray(test_img_1),numpy.asarray(test_img_2)])

prob[prob>= 0.5] = 1
prob[prob<0.5] = 0
print (prob)

test_eval = siamese.evaluate(x=[numpy.asarray(test_img_1),numpy.asarray(test_img_2)],y=numpy.asarray(test_label),verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


#For testing hidden data
#initialise image array
t_x=[]
t_y=[]
WIDTH = 128
HEIGHT = 128

AND_hidden_Dataset_path="/Users/sunitapattanayak/Documents/AMLPROJECT/AND_dataset/Dataset[Without-Features]/DL/DLTestData"
onlyfiles = [ f for f in listdir(AND_hidden_Dataset_path) if isfile(join(AND_hidden_Dataset_path,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for x in range(0, len(onlyfiles)):
    image_path = os.path.join(AND_hidden_Dataset_path, onlyfiles[x])
    if os.path.exists(image_path):
        images[x] = cv2.imread( image_path )
        t_x.append(cv2.resize(images[x], (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        t_y.append(basename(onlyfiles[x]))

t_img_1=[]
t_img_2=[]
t_label=[]

import pandas as pd

col=("FirstImage","SecondImage")
df=pd.DataFrame(columns=col)


import csv
with open('/Users/sunitapattanayak/Downloads/DLTestPairs.csv') as csvfile:
     rows = csv.reader(csvfile)
     next(rows)
     c=0
     for row in rows:
         img1=row[1]
         img2=row[2]
         df.loc[c]=[img1,img2]
         c+=1
         index1=t_y.index(img1)
         index2=t_y.index(img2)
         t_img_1.append(t_x[index1])
         t_img_2.append(t_x[index2])
                    
prob = siamese.predict([numpy.asarray(t_img_1),numpy.asarray(t_img_2)])
prob[prob>= 0.5] = 1
prob[prob<0.5] = 0
print (prob)
#for finding the same or different column with 0 or 1
for l in prob:
    print(l)
    if(list(l)==[1.,0.]):
        t_label.append(0)
    else:
        t_label.append(1)

df['SameOrDifferent']=t_label
#write to the output csv file
csvfile="CNNTestOutput.csv"
df.to_csv(csvfile, sep=',',encoding='utf-8')















