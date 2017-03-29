import sys
import csv
import numpy as np

# argv : raw_data_X raw_data_Y test_X

global data_item
data_item = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country']
global data_index 
data_index = dict(zip(data_item,[ int(i) for i in range(0,len(data_item),1)]))
global data_set
global sum_data 
sum_data = dict()
global class_data
global pro_data
global std
class_data = dict()
data_feature = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country']
data_set = data_feature

def Train_Input(fileX,fileY):
	global data_index
	global data_item
	global sum_data
	global class_data
	global pro_data
	global std
	train_data = [[],[]]

	train_X = open(fileX,'r')
	raw_data_X = list(csv.reader(train_X))
	counter = 0
	data = [[],[]]
	train_Y = open(fileY,'r')
	for y in csv.reader(train_Y):
		data[int(y[0])].append(raw_data_X[counter])
		counter = counter + 1
	for i in range(0,2):
		for datarow in data[i]:
			for item in datarow[:-1]:
				item = item.replace(" ","")
				if item.isdigit() == False:
					sum_data[item] = sum_data.get(item,0) + 1
					if i == 0:
						class_data[item] = 0.0
					if i == 1:
						class_data[item] = class_data.get(item,0) + 1
	pro_data = dict((item.replace(" ",""),float(class_data[item]/sum_data[item])) for item in sum_data.keys())
	for i in range(0,2):
		for datarow in data[i]:
			row = []
			for item in datarow[:-1]:
				item = item.replace(" ","")
				if item.isdigit() == False:
					k = pro_data[item]
					row.append(k)
				else :
					row.append(float(item))
			train_data[i].append(row)
	#standard
	tmp = np.array(train_data[0]+train_data[1],dtype="float")
	std = np.std(tmp,axis=0)

	train_data_std = [[],[]]
	for i in range(0,2):
		for datarow in train_data[i]:
			row = []
			for j in range(0,len(datarow)):
				row.append(float(datarow[j])/std[j])
			train_data_std[i].append(row)

	train_X.close()
	train_Y.close()
	return train_data_std

def Test_Input(filename,p):
	global pro_data
	global std
	test_data = []
	row = []

	testfile = open(filename,'r')
	for datarow in csv.reader(testfile):
		row = []
		for i in range(0,len(datarow)):
			item = datarow[i]
			item = item.replace(" ","")
			if item.isdigit() == False:
				k = 0.0
				if sum_data.get(item,0) == 0:
					k = p/std[i]
				else :
					k = pro_data[item]/std[i]
				row.append(k)
			else :
				row.append(float(item)/std[i])
		test_data.append(row)
	testfile.close()
	return test_data

def Extract(raw_data):
	global data_item
	global data_index
	global data_continuous
	global data_set

	feature = []
	for datarow in raw_data:
		vector = []
		for item in data_item:
			if item in data_set:
				vector.append(datarow[data_index[item]])
		feature.append(vector)
	return feature

def Output(feature,filename):
	global data_set

	filevar = open(filename,'w')
	wf = csv.writer(filevar)
	wf.writerow(data_set)
	for datarow in feature:
		wf.writerow(datarow)
	filevar.close()

#main
train_data = Train_Input(sys.argv[1],sys.argv[2])
feature = []
for datarow in train_data:
	feature.append(Extract(datarow))
Output(feature,"generative_model.csv")
test_data = Test_Input(sys.argv[3],float(len(train_data[1]))/(len(train_data[0])+len(train_data[1])))
test_feature = Extract(test_data)
Output(test_feature,"generative_test.csv")