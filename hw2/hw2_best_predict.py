import sys
import csv
import numpy as np

def Sigmoid(z):
	return 1.0/(1.0+np.exp(-1.0*z))

def Param_Input(filename):
	param_file = open(filename,'r')
	data_array = list(csv.reader(param_file))
	w = [eval(v) for v in data_array[0]]
	b = float(data_array[1][0][1:-1])
	return w,b

def Test_Input(filename):
	test_data = []

	test_file = open(filename,'r')
	data_array = list(csv.reader(test_file))
	for i in data_array[1]:
		data = eval(i)
		row = []
		for k in data:
			row.append(float(k))
		test_data.append(row)
	return test_data

def Predict(test_data,w,b,filename):
	result_file = open(filename,'w')
	wf = csv.writer(result_file)
	wf.writerow(['id','label'])
	id_num = 1
	prob = 0.0
	pred = 0

	for pre_data in test_data:
		prob = Sigmoid(np.dot(np.array(pre_data,dtype="float")[np.newaxis],w) + b)
		pred = round(prob,0)
		# print prob,pred
		wf.writerow([id_num,int(pred)])
		id_num = id_num + 1
	result_file.close()
	return 


#main
w,b = Param_Input("best_params.csv")
test_data = Test_Input("best_test.csv")
Predict(test_data,w,b,sys.argv[1])
