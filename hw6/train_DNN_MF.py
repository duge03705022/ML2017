import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
import h5py
import csv
import numpy as np
import sys
import random

def movie_input(movie_path):
	
	raw_data = [line.split("::") for line in (open(movie_path,'r',encoding="Windows-1252").read()).split("\n")][1:-1]
	movieid_mapping = dict((int(raw_data[i][0]),i) for i in range(0,len(raw_data)))
	n_movie = len(raw_data)

	typeid_list = []
	for i in range(0,len(raw_data)):
		type_str = raw_data[i][2]
		if type_str not in typeid_list:
			typeid_list.append(type_str)

	typeid_list = sorted(typeid_list)

	n_type = len(typeid_list)
	typeid_mapping = dict((typeid_list[i],i) for i in range(0,len(typeid_list)))

	movie_vecs = np.array([typeid_mapping[line[2]] for line in raw_data])
	return (n_movie,movieid_mapping),n_type,movie_vecs

def user_input(user_path):

	raw_data = [line.split("::") for line in (open(user_path,'r',encoding="Windows-1252").read().replace("F","1").replace("M","0")).split("\n")][1:-1]
	userid_mapping = dict((int(raw_data[i][0]),i) for i in range(0,len(raw_data)))
	n_user = len(raw_data)

	user_list = []
	for i in range(0,len(raw_data)):
		if raw_data[i][1:] not in user_list:
			user_list.append(raw_data[i][1:])

	n_feature = len(user_list)
	user_mapping = dict((str(user_list[i]),i) for i in range(0,n_feature))

	user_features = np.array([user_mapping[str(line[1:])] for line in raw_data])
	
	return (n_user,userid_mapping),(n_feature,user_features)

def train_input(train_path,n_movie,n_user,movieid_mapping,userid_mapping,split_ratio,movie_vecs,user_features):

	raw_data = list(csv.reader(open(train_path,'r')))[1:]
	split_num = int(len(raw_data)*split_ratio) + 850
	random.shuffle(raw_data)
	
	val = raw_data[-split_num:]
	train = raw_data[:-split_num]
	
	train_X = [line[:-1] for line in train]
	train_Y = [line[-1] for line in train]
	val_X = [line[:-1] for line in val]
	val_Y = [line[-1] for line in val]

	train_user = [userid_mapping[int(line[1])] for line in train_X]
	train_movie = [movieid_mapping[int(line[2])] for line in train_X]

	val_user = [userid_mapping[int(line[1])] for line in val_X]
	val_movie = [movieid_mapping[int(line[2])] for line in val_X]

	return (np.array(train_movie),np.array(train_user)),(np.array(val_movie),np.array(val_user)),(np.array(train_Y),np.array(val_Y))

def test_input(test_path,movieid_mapping,userid_mapping,movie_vecs,user_features):

	raw_data = list(csv.reader(open(test_path,'r')))[1:]

	test_vecs = [[int(line[0]), userid_mapping[int(line[1])],movieid_mapping[int(line[2])]] for line in raw_data]
	test_movie = [line[2] for line in test_vecs]
	test_user = [line[1] for line in test_vecs]

	return test_vecs,np.array(test_movie),np.array(test_user)

def rmse(y_true,y_pred):
	return K.sqrt(K.mean(((y_pred - y_true)**2)))

def wrong_rmse(y_true,y_pred):
	return K.mean(K.sqrt(((y_pred - y_true)**2)))

### parameter
data_path = sys.argv[1]
out_path = sys.argv[2]
split_ratio = 0.3
input_dims = 25
dense_num = 100
nb_epoch = 10000
batch_size = 1000

### input
print("Get movie, user data")
(n_movie,movieid_mapping),n_type,movie_vecs = movie_input(data_path+"movies.csv")
(n_user,userid_mapping),(n_feature,user_features) = user_input(data_path+"users.csv")
print("Get training data")
(train_movie,train_user),(val_movie,val_user),(Y_train,Y_val) = train_input("train.csv",n_movie,n_user,movieid_mapping,userid_mapping,split_ratio,movie_vecs,user_features)
print("Get testing data")
test_vecs,test_movie,test_user = test_input(data_path+"test.csv",movieid_mapping,userid_mapping,movie_vecs,user_features)

#build model
movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(
	keras.layers.Embedding(n_movie, input_dims , embeddings_initializer = 'RandomNormal')(movie_input))

user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(
	keras.layers.Embedding(n_user, input_dims , embeddings_initializer = 'RandomNormal')(user_input))

user_bias = keras.layers.Flatten()(
	keras.layers.Embedding(n_user, 1 , embeddings_initializer = 'zeros')(user_input))

movie_bias = keras.layers.Flatten()(
	keras.layers.Embedding(n_movie, 1 , embeddings_initializer = 'zeros')(movie_input))

r = keras.layers.Dot(axes=1)([user_vec,movie_vec])
r = keras.layers.Add()([r,user_bias,movie_bias])

model = kmodels.Model([movie_input, user_input], r)
model.summary()
model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=[wrong_rmse,rmse])

earlystopping = EarlyStopping(monitor='val_rmse', patience = 20, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath="%s.hdf5" % out_path,
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_rmse',
                             mode='min')

hist = model.fit([train_movie,train_user], Y_train, 
                 validation_data=([val_movie,val_user], Y_val),
                 epochs=nb_epoch, 
                 batch_size=batch_size,
                 callbacks=[earlystopping,checkpoint])

model.load_weights("%s.hdf5" % out_path)

score = model.evaluate([val_movie,val_user], Y_val)

model.save("model/%s.h5" % str(score[2]))

Y_pred = model.predict([test_movie,test_user],verbose=1)

out = open(out_path,'w')
csvw = csv.writer(out)
csvw.writerow(["TestDataID","Rating"])
for i in range(0,len(test_vecs)):
	csvw.writerow([test_vecs[i][0],max(1.0,min(Y_pred[i][0],5.0))])
out.close()