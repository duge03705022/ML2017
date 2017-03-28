import sys
import csv
import numpy as np

# argv : raw_data_X raw_data_Y test_X

global data_item
global data_continuous 
data_continuous = ['age','fnlwgt','capital_gain','capital_loss','hours_per_week']
global data_index
global data_set

def Input(fileX,fileY):
	global data_index
	global data_item
	train_data = []

	
	train_X = open(fileX,'r')
	data_array = list(csv.reader(train_X))
	data_item = [s.replace('-','_') for s in data_array[0]]
	data_index = dict(zip(data_item,[i for i in range(0,len(data_item),1)]))
	train_data = data_array[1:]
	if fileY == "0":
		return train_data
	counter = 0
	data = [[],[]]
	train_Y = open(fileY,'r')
	for y in csv.reader(train_Y):
		data[int(y[0])].append(train_data[counter])
		counter = counter + 1
	return data

def Extract(raw_data):
	global data_item
	global data_index
	global data_continuous
	global data_set

	feature = []
	data_set = data_continuous
	# for i in range(0,2):
	for datarow in raw_data:
		vector = []
		for item in data_item:
			if item in data_set:
				vector.append(datarow[data_index[item]])
		feature.append(vector)
	return feature

def Output(feature,filename):
	global data_item

	filevar = open(filename,'w')
	wf = csv.writer(filevar)
	wf.writerow(data_set)
	for datarow in feature:
		wf.writerow(datarow)
	filevar.close()

#main
train_data = Input(sys.argv[1],sys.argv[2])
feature = []
for datarow in train_data:
	feature.append(Extract(datarow))
Output(feature,"generative_model.csv")
test_data = Input(sys.argv[3],"0")
test_feature = Extract(test_data)
Output(test_feature,"generative_test.csv")