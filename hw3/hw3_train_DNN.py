import sys
import csv
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "device=gpu0"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import *
#categorical_crossentropy


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
	x_train = np.array(x_train[:train_num])
	y_train = np.array(y_train[:train_num])

	x_train = x_train.reshape(len(x_train),48,48,1)
	x_test = x_test.reshape(len(x_test),48,48,1)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, 7)
	y_test = np_utils.to_categorical(y_test, 7)
	x_train = x_train/255
	x_test = x_test/255
	#x_test=np.random.normal(x_test)
	return (x_train, y_train), (x_test, y_test)


(x_train,y_train),(x_test,y_test)=load_data(sys.argv[1])

model = Sequential()
model.add(Dense(units=25 ,input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))
model.add(Dense(units=50))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Dense(units=100))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Dense(units=200))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=800))
model.add(Dense(units=400))
model.add(Dense(units=200,activation='relu'))
model.add(Dense(units=7,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
batch_size = 200
epoch = 20
history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch)

print(history.history.keys())
print history

score = model.evaluate(x_train,y_train)
print '\nLoss:', score[0]
print '\nTrain Acc:', score[1]
score = model.evaluate(x_test,y_test)
print '\nLoss:', score[0]
print '\nTest Acc:', score[1]
model.save("model/DNN_"+str(score[1])+".h5")