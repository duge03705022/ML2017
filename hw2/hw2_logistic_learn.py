import time

import sys
import csv
import numpy as np

begin = time.clock();
global data_item
global data_index
global data_item_len
global w
global b
global learn_rate
global loss
global dist

def Sigmoid(z):
	return 1.0/(1.0+np.exp(-1.0*z))

def Train_Input(filename):
	global data_item
	global data_index
	global data_item_len
	train_data = []

	train_file = open(filename,'r')
	data_array = list(csv.reader(train_file))
	data_item = data_array[0]
	data_item_len = len(data_item)
	data_index = dict(zip(data_item,[i for i in range(0,data_item_len,1)]))
	data_array[1] = [eval(v) for v in data_array[1]]
	for i in data_array[1]:
		train_data.append([float(k) for k in i])
	return train_data

def Init(w_inv,b_inv):
	global w
	global b
	global learn_rate
	global loss
	global dist
	global data_item_len

	b = b_inv
	w = np.array(w_inv).reshape(data_item_len,1)
	learn_rate = 0.5
	loss = 100000000
	dist = 0.1
	return

def Learning(train_data):
	global w
	global b
	global learn_rate
	global loss
	global dist
	global data_item_len
	avoid_bound = 0.00000001

	last_loss = 0
	while loss > 40 and (loss - last_loss > 0 or last_loss - loss > dist) :
		if loss > last_loss:
			learn_rate = learn_rate/2
		last_loss = loss
		b_grad = 0
		w_grad = np.full((data_item_len,),0.0)
		b_lr = 0
		w_lr = np.full((data_item_len,),0.0)
		prob = 0.0
		loss_sum = 0.0
		for person_data in train_data:
			y = float(person_data[len(person_data)-1])
			vector = person_data[:-1]
			prob = Sigmoid(np.dot(np.array(vector)[np.newaxis],w) + b)
			loss_sum = loss_sum - 1.0 * (y * np.log((prob+avoid_bound) * (1 - avoid_bound)) + (1 - y) * np.log((1-prob+avoid_bound)*(1 - avoid_bound)) )
			for i in range(0,data_item_len):
				w_grad[i] = w_grad[i] - (y - prob)*vector[i]
				w_lr[i] = w_lr[i] + w_grad[i]**2
				w[i] = w[i] - learn_rate/np.sqrt(w_lr[i]+avoid_bound) * w_grad[i]
			b_grad = b_grad - (y - prob)
			b_lr = b_lr + b_grad**2.0
			b = b - learn_rate/np.sqrt(b_lr+avoid_bound) * b_grad
		loss = loss_sum
		print (last_loss,loss)
	return

def Output(filename):
	global w
	global b

	param_file = open(filename,'w')
	wf = csv.writer(param_file)
	wf.writerow(w)
	wf.writerow(b)
	param_file.close()
	return

#main
train_data = Train_Input("logistic_model.csv")
w_inv = []
for i in range(0,data_item_len):
	w_inv.append(1.0/data_item_len)
b_inv = 0.0
Init(w_inv,b_inv)
Learning(train_data)
Output("logistic_params.csv")
print time.clock()-begin
# p,mu,com_cov=Learning(train_data)
# test_data = Test_Input("generative_test.csv")
# Predict(test_data,p,mu,com_cov,sys.argv[1])
