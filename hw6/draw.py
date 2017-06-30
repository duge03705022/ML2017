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
from matplotlib import pyplot as plt
from tsne import bh_sne

def movie_input(movie_path):
	
	raw_data = [line.split("::") for line in (open(movie_path,'r',encoding="Windows-1252").read()).split("\n")][1:-1]
	movieid_mapping = dict((int(raw_data[i][0]),i) for i in range(0,len(raw_data)))
	n_movie = len(raw_data)

	typeid_list = []
	for i in range(0,len(raw_data)):
		type_str = raw_data[i][2]
		type_list = type_str.split("|")
		for tp in type_list:
			if tp not in typeid_list:
				typeid_list.append(tp)

	typeid_list = sorted(typeid_list)

	n_type = len(typeid_list)
	typeid_mapping = {'Action':0, 'Adventure':1, 'Animation':0, "Children's":0, 'Comedy':0, 'Crime':1,
						  'Documentary':1, 'Drama':1, 'Fantasy':0, 'Film-Noir':1, 'Horror':1, 'Musical':0,
						  'Mystery':1, 'Romance':0, 'Sci-Fi':1, 'Thriller':1, 'War':0, 'Western':0}

	movie_vecs = []
	for i in range(0,len(raw_data)):
		type_str = raw_data[i][2]
		type_list = type_str.split("|")
		# vec = [1 if (typeid_list[i] in type_list) else 0 for i in range(0,n_type)]
		movie_vecs.append(typeid_mapping[type_list[0]])

	movie_vecs = np.array(movie_vecs)
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

def rmse(y_true,y_pred):
	return K.sqrt(K.mean(((y_pred - y_true)**2)))

def wrong_rmse(y_true,y_pred):
	return K.mean(K.sqrt(((y_pred - y_true)**2)))

def get_embedding(model):
	user_emb = np.array(model.layers[2].get_weights()).squeeze()
	movie_emb = np.array(model.layers[3].get_weights()).squeeze()
	return user_emb,movie_emb

def draw(x,y):
	y = np.array(y)
	x = np.array(x,dtype=np.float64)

	vis_data = bh_sne(x)

	vis_x = vis_data[:,0]
	vis_y = vis_data[:,1]

	cm = plt.cm.get_cmap('RdYlBu')
	sc = plt.scatter(vis_x,vis_y,c=y,cmap=cm)
	plt.colorbar(sc)
	plt.show()
	return

### parameter
data_path = sys.argv[1]
model_path = sys.argv[2]
split_ratio = 0.2

### input
print("Get movie, user data")
(n_movie,movieid_mapping),n_type,movie_vecs = movie_input(data_path+"movies.csv")
(n_user,userid_mapping),(n_feature,user_features) = user_input(data_path+"users.csv")

print("Get training data")
(train_movie,train_user),(val_movie,val_user),(Y_train,Y_val) = train_input("train.csv",n_movie,n_user,movieid_mapping,userid_mapping,split_ratio,movie_vecs,user_features)

model = load_model(model_path,custom_objects={"rmse":rmse,"wrong_rmse":wrong_rmse})
model.summary()
# adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)

user_emb,movie_emb = get_embedding(model)
draw(movie_emb,movie_vecs)
