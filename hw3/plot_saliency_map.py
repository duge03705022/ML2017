import os
import sys
import csv
import pickle
import argparse
import tensorflow as tf 
from keras.models import load_model
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from vis.utils import utils
from vis.visualization import visualize_saliency

# python plot_saliency_map.py model.h5
# 
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

emotion_classifier = load_model("model/"+sys.argv[1])
layer_idx = [idx for idx, layer in enumerate(emotion_classifier.layers) if layer.name == "dense_3"][0]
private_pixels = load_data("test.csv")
input_img = emotion_classifier.input
img_ids = []

for i in xrange(1,20):
	img_ids.append(i)

for idx in img_ids:
	
	val_proba = emotion_classifier.predict(private_pixels[idx:idx+1])
	pred = val_proba.argmax(axis=-1)
	target = K.mean(emotion_classifier.output[:, pred])
	grads = K.gradients(target, input_img)[0]
	fn = K.function([input_img, K.learning_phase()], [grads])
	
	heatmap = visualize_saliency(emotion_classifier, layer_idx, pred, private_pixels[idx])

	thres = 0.5
	see = private_pixels[idx].reshape(48, 48)
	mp = np.mean(see)
	for i in range(0,48):
		for j in range(0,48):
			if np.mean(heatmap[i][j]) < thres:
				see[i][j] = mp

	if not os.path.exists("model_data/saliency_map/"+sys.argv[1]):
		os.makedirs("model_data/saliency_map/"+sys.argv[1])

	plt.figure()
	plt.imshow(heatmap, cmap=plt.cm.jet)
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig("model_data/saliency_map/"+sys.argv[1]+"/"+str(idx)+".png", dpi=100)

	plt.figure()
	plt.imshow(see,cmap='gray')
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig("model_data/saliency_map/"+sys.argv[1]+"/"+str(idx)+"_mask.png", dpi=100)
	plt.close()