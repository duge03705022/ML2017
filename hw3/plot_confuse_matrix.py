from keras.models import load_model
from sklearn.metrics import confusion_matrix
# from marcos import exp_dir
import csv
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import itertools
from keras.utils import np_utils

# python plot_confuse_matrix.py model.h5

def load_data(fileX):
	train_raw = list(csv.reader(open(fileX,'r')))
	train_num = 24000
	x_train = []
	y_train = []
	x_test = []
	y_test = []

	print "Begin loading..."
	for item in train_raw[1:]:
		y_train.append(int(item[0]))
		x_train.append([int(i) for i in item[1].split(" ")])
	print "Loading completed"
	
	x_test = np.array(x_train[train_num:])
	y_test = np.array(y_train[train_num:])
	
	x_test = x_test.reshape(len(x_test),48,48,1)
	x_test = x_test.astype('float32')
	# convert class vectors to binary class matrices
	y_test = np_utils.to_categorical(y_test, 7)
	x_test = x_test/255
	#x_test=np.random.normal(x_test)
	return (x_train, y_train), (x_test, y_test)

def get_label(y_test):
	res = []
	for item in y_test:
		for i in range(0,len(item)):
			if item[i]==1.0:
				res.append(i)
	return res

def plot_confusion_matrix(cm, classes,
						  title='Confusion matrix',
						  cmap=plt.cm.jet):
    
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

modelname = sys.argv[1]
emotion_classifier = load_model("model/"+modelname)
emotion_classifier.summary()
(x_train,y_train),(x_test,y_test)=load_data("data/train.csv")
np.set_printoptions(precision=2)
predictions = emotion_classifier.predict_classes(x_test)
label = get_label(y_test)
conf_mat = confusion_matrix(label,predictions)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.savefig("model_data/confuse_matrix/"+modelname+".png")