import sys
import csv
import numpy as np

global data_item
global data_index

def Train_Input(filename):
	global data_item
	global data_index
	train_data = [[],[]]

	train_file = open(filename,'r')
	data_array = list(csv.reader(train_file))
	data_item = data_array[0]
	data_index = dict(zip(data_item,[i for i in range(0,len(data_item),1)]))
	data_array[1] = [eval(v) for v in data_array[1]]
	data_array[2] = [eval(v) for v in data_array[2]]
	for i in data_array[1]:
		train_data[0].append([float(k) for k in i])
	for i in data_array[2]:
		train_data[1].append([float(k) for k in i])
	return train_data

def Test_Input(filename):
	test_file = open(filename,'r')
	data_array = list(csv.reader(test_file))
	test_data = data_array[1:]
	return test_data

def Learning(train_data):
	p = [[],[]]
	mu = [[],[]]
	mu_sub = [[],[]]
	cov_mat = [[],[]]
	cov = [[],[]]

	dim = len(train_data[0][0])
	p[0] = float(len(train_data[0]))/(len(train_data[0])+len(train_data[1]))
	p[1] = float(len(train_data[1]))/(len(train_data[0])+len(train_data[1]))
	mu[0] = np.mean(np.array(train_data[0],dtype="float"),axis=0)
	mu[1] = np.mean(np.array(train_data[1],dtype="float"),axis=0)
	mu_sub[0] = [np.array(v,dtype="float")[np.newaxis]-mu[0] for v in train_data[0]]
	mu_sub[1] = [np.array(v,dtype="float")[np.newaxis]-mu[1] for v in train_data[1]]
	cov_mat[0] = [np.dot(np.transpose(v),v)/len(train_data[0]) for v in mu_sub[0]]
	cov[0] = np.zeros((dim,dim),dtype="float")
	for mat in cov_mat[0]:
		cov[0] = np.add(mat,cov[0])
	cov_mat[1] = [np.dot(np.transpose(v),v)/len(train_data[1]) for v in mu_sub[1]]
	cov[1] = np.zeros((dim,dim),dtype="float")
	for mat in cov_mat[1]:
		cov[1] = np.add(mat,cov[1])
	com_cov = np.mat(p[0]*cov[0]+p[1]*cov[1])
	return p,mu,com_cov

def Predict(test_data,p,mu,com_cov,filename):
	result_file = open(filename,'w')
	wf = csv.writer(result_file)
	wf.writerow(['id','label'])
	id_num = 1

	for pre_data in test_data[1:]:
		class_num = -1
		p_class = [0.0,0.0]
		pred = 0.0

		v = np.array(pre_data,dtype="float")[np.newaxis]
		p_class[0] = np.exp(-0.5*np.dot(np.dot((v-mu[0]),com_cov.I),np.transpose(v-mu[0])))*p[0]
		p_class[1] = np.exp(-0.5*np.dot(np.dot((v-mu[1]),com_cov.I),np.transpose(v-mu[1])))*p[1]
		pred = p_class[0]/(p_class[0]+p_class[1])

		if pred > 0.5 :
			class_num = 0
		else :
			class_num = 1
		wf.writerow([id_num,class_num])
		id_num = id_num + 1
	return 


#main
train_data = Train_Input("generative_model.csv")
p,mu,com_cov=Learning(train_data)
test_data = Test_Input("generative_test.csv")
Predict(test_data,p,mu,com_cov,sys.argv[1])
