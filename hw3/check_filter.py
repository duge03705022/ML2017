import matplotlib.pyplot as plt
import math, csv, random, copy
import numpy as np 
import keras
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
import os
import sys
from keras import applications
from keras import backend as K
from scipy.misc import imsave
from PIL import Image

# python check_filter.py model.h5 conv2d_1

def load_data(fileX):
    test_raw = list(csv.reader(open(fileX,'r')))
    x_test = []
    
    print "Begin loading..."
    for item in test_raw[1:]:
        x_test.append([int(i) for i in item[1].split(" ")])
    print "Loading completed"
    
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(x_test),48,48,1)
    x_test = x_test.astype('float32')
    x_test = x_test/255
    return x_test

name = sys.argv[1]
model = load_model('model/' + name)

pics = load_data("test.csv")
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = sys.argv[2]
layer_output = layer_dict[layer_name].output


nb_filter = 24
fig = plt.figure(figsize=(14, 8))
fig2 = plt.figure(figsize=(14, 8))
for i in range(nb_filter):
    loss = K.mean(layer_output[:, :, :, i])
    input_img = model.input
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 0.00001)


    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])
    input_img_data = np.random.random((1, 48, 48, 1)) # random noise

    # run gradient ascent for 100 steps
    for step in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    img = img.reshape(48,48)
    
    ax = fig.add_subplot(nb_filter/8, 8, i+1)
    ax.imshow(img, cmap='Blues')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()

    input_img_data = pics[1].reshape(1,48,48,1)
    for step in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    img = img.reshape(48,48)
    
    ax = fig2.add_subplot(nb_filter/8, 8, i+1)
    ax.imshow(img, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()

# fig.show()
fig.suptitle(name)
fig.savefig("model_data/filter/"+name+"_"+layer_name+".png")
fig2.suptitle(name+"with img")
fig2.savefig("model_data/filter/"+name+"_"+layer_name+"_with_img.png")