import sys
import csv
import numpy as np

# argv : raw_data_X raw_data_Y test_X

global data_item
global data_index 
global data_set
global data_continuous
global std
global mean
data_continuous = ["age","fnlwgt","capital_gain","capital_loss","hours_per_week"]

def Train_Input(fileX,fileY):
	global data_index
	global data_item
	global data_continuous
	global data_set
	global std
	global mean
	train_data = []

	train_X = open(fileX,'r')
	raw_data_X = list(csv.reader(train_X))
	data_item = raw_data_X[0]
	data_index = dict(zip(data_item,[ int(i) for i in range(0,len(data_item),1)]))
	data_set = data_item
	counter = 1
	train_data = raw_data_X[1:]

	train_Y = open(fileY,'r')
	raw_data_Y = list(csv.reader(train_Y))
	for i in range(0,len(train_data)):
		for j in range(0,len(train_data[i])):
			train_data[i][j]=float(train_data[i][j])
		train_data[i].append(float(raw_data_Y[i][0]))
	
	#standard
	tmp = np.array(train_data,dtype="float")
	std = np.std(tmp,axis=0)
	mean = np.mean(tmp,axis=0)

	train_data_std = []
	for datarow in train_data:
		row = []
		for j in range(0,len(datarow)-1):
			if data_item[j] in data_continuous:
				row.append((float(datarow[j])-mean[j])/std[j])
			else :
				row.append(float(datarow[j]))
		row.append(datarow[len(datarow)-1])
		train_data_std.append(row)

	train_X.close()
	train_Y.close()
	return train_data_std

def Test_Input(filename):
	global data_continuous
	global data_item
	global std
	global mean
	test_data = []
	row = []

	testfile = open(filename,'r')
	test_input = list(csv.reader(testfile))
	for datarow in test_input[1:]:
		row = []
		for i in range(0,len(datarow)):
			if data_item[i] in data_continuous:
				row.append((float(datarow[i])-mean[i])/std[i])
			else :
				row.append(float(datarow[i]))
		test_data.append(row)
	testfile.close()
	return test_data

def Extract(raw_data,types):
	global data_item
	global data_index
	global data_continuous
	global data_set
	# data_delete = ['fnlwgt']

	# for item in data_delete:
	# 	data_set.remove(item)
	feature = []
	for datarow in raw_data:
		vector = []
		for item in data_item:
			if item in data_set:
				vector.append(datarow[data_index[item]])
		if types == 1:
			vector.append(datarow[len(datarow)-1])
		feature.append(vector)
	return feature

def Output(feature,filename):
	global data_set

	filevar = open(filename,'w')
	wf = csv.writer(filevar)
	wf.writerow(data_set)
	wf.writerow(feature)
	filevar.close()

#main
train_data = Train_Input(sys.argv[1],sys.argv[2])
feature = Extract(train_data,1)
Output(feature,"logistic_model.csv")
test_data = Test_Input(sys.argv[3])
test_feature = Extract(test_data,0)
Output(test_feature,"logistic_test.csv")