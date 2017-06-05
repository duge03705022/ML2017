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
from keras.models import load_model

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

	# movie_vecs = []
	# for i in range(0,len(raw_data)):
	# 	type_str = raw_data[i][2]
	# 	type_list = type_str.split("|")
	# 	vec = [1 if (typeid_list[i] in type_list) else 0 for i in range(0,n_type)]
	# 	movie_vecs.append(vec)

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
	# train_X = []
	# train_Y = []
	# val_X = []
	# val_Y = []
	val = raw_data[-split_num:]
	train = raw_data[:-split_num]
	
	# for i in range(0,len(raw_data)):
	# 	if i in ids:
	# 		val_X.append(raw_data[i][:-1])
	# 		val_Y.append(raw_data[i][-1])
	# 	else :
	# 		train_X.append(raw_data[i][:-1])
	# 		train_Y.append(raw_data[i][-1])
	
	# for index in ids:
	# 	if index < (len(raw_data)):
	# 		val.append(raw_data[index])
	# 		train.remove(raw_data[index])
	train_X = [line[:-1] for line in train]
	train_Y = [line[-1] for line in train]
	val_X = [line[:-1] for line in val]
	val_Y = [line[-1] for line in val]

	train_user = [userid_mapping[int(line[1])] for line in train_X]
	train_movie = [movieid_mapping[int(line[2])] for line in train_X]
	# Y_train = [[ 1 if x == int(line) else 0 for x in range(1,6)] for line in train_Y]

	val_user = [userid_mapping[int(line[1])] for line in val_X]
	val_movie = [movieid_mapping[int(line[2])] for line in val_X]
	# Y_val = [[ 1 if x == int(line) else 0 for x in range(1,6)] for line in val_Y]

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
model_name = sys.argv[3]
split_ratio = 0.2

### input
print("Get movie, user data")
(n_movie,movieid_mapping),n_type,movie_vecs = movie_input(data_path+"movies.csv")
(n_user,userid_mapping),(n_feature,user_features) = user_input(data_path+"users.csv")
print("Get training data")
(train_movie,train_user),(val_movie,val_user),(Y_train,Y_val) = train_input("train.csv",n_movie,n_user,movieid_mapping,userid_mapping,split_ratio,movie_vecs,user_features)
print("Get testing data")
test_vecs,test_movie,test_user = test_input(data_path+"test.csv",movieid_mapping,userid_mapping,movie_vecs,user_features)

model = load_model("model/%s.h5" % model_name,custom_objects={"rmse":rmse})
model.summary()
# adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)

Y_pred = model.predict([test_movie,test_user],verbose=1)
Y_pred = K.eval(K.round(Y_pred))
out = open(out_path,'w')
csvw = csv.writer(out)
csvw.writerow(["TestDataID","Rating"])
for i in range(0,len(test_vecs)):
	csvw.writerow([test_vecs[i][0],max(1.0,min(Y_pred[i][0],5.0))])
out.close()