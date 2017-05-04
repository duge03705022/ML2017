import time
import csv
import numpy as np
import os
import sys
os.environ["THEANO_FLAGS"] = "device=gpu0"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model
#categorical_crossentropy

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

model_name = "model/best_model.h5"
x_test=load_data(sys.argv[1])
model = load_model(model_name)
result = model.predict(x_test)

# result_file = open("result/result_"+model_name+".csv","w")
result_file = open(sys.argv[2],"w")
rw = csv.writer(result_file)
rw.writerow(["id","label"])
count = 0
for item in result:
	ma = max(item)
	index = -1
	for i in range(0,len(item)):
		if item[i] == ma :
			index = i
	rw.writerow([count,index])
	count = count + 1
result_file.close()
