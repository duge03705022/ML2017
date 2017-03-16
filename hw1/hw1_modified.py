#import pdb; pdb.set_trace()
import time
#-----------------for test(have to delete when completed)---------------
import sys
import csv
import numpy as np

#run : python data_file test_file result_file model_file profm_id bool_init
progm_id = sys.argv[5]
model_file = sys.argv[4]

begin = time.clock();
data_item = ["AMB_TEMP","CH4","CO","NMHC","NO","NO2","NOx","O3","PM10","PM2.5","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED","WS_HR"]
learn_rate = 0.1
loss = 100000000
dist = 0
b_iv = 0
w_iv = 0

#Init
def Init():
	global data_item
	global b
	global w
	global learn_rate
	global loss
	global dist
	global b_iv
	global w_iv

	b_iv = 0.1
	w_iv = 0.1
	data_item = ["PM2.5"]
	b = dict((k,b_iv) for k in data_item)
	w = dict((k,np.full((9,),w_iv)) for k in data_item)
	learn_rate = 0.1
	loss = 100000000
	dist = 0.000001
	return True

#Import_model
#Return : (Output : ["model_"+sys.argv[5]].csv)
def Import_model():
	global data_item
	global b
	global w
	global learn_rate
	global dist
	global b_iv
	global w_iv

	input_file = open(model_file,'r')
	for row in csv.reader(input_file):
		if row[0]=="data_item":
			data_item=eval(row[1])
		if row[0]=="b_iv":
			b_iv = float(row[1])
			b=dict((k,b_iv) for k in data_item)
		if row[0]=="w_iv":
			w_iv = float(row[1])
			w=dict((k,np.full((9,),w_iv)) for k in data_item)
		if row[0]=="learn_rate":
			learn_rate=float(row[1])
		if row[0]=="dist":
			dist=float(row[1])
	return True

#Export_model
#Return : (Output : ["model_"+sys.argv[5]].csv)
def Export_model(filetype):
	global data_item
	global b
	global w
	global learn_rate
	global loss

	filename = "log/"+filetype+progm_id+".csv"
	output_file = open(filename,'w')
	wf = csv.writer(output_file)
	wf.writerow(["data_item",data_item])
	wf.writerow(["b_iv",b_iv])
	wf.writerow(["w_iv",w_iv])
	wf.writerow(["learn_rate",learn_rate])
	wf.writerow(["dist",dist])
	if filetype == "resultparam_":
		wf.writerow(["loss",loss])
	return

#Training Data Input
#Return : train_data[day][item][item_data]
def Train_Input():
	train_data = []
	day_data = []
	train_file = open(sys.argv[1],'r')
	last = ""
	for row in csv.reader(train_file):
		row = [s.replace("NR","0") for s in row]
		for i in range(3,len(row),1):
			row[i] = float(row[i])
		if last == row[0]:
			day_data.append(row)
		else :
			if day_data != [] :
				train_data.append(day_data)
			day_data = []
			day_data.append(row)
			last = row[0]
	train_file.close()
	train_data.append(day_data)
	return train_data

#Training Test Input
#Return : test_data[id][item][item_data]
def Test_Input():
	test_data = []
	id_data = []
	test_file = open(sys.argv[2],'r')
	last = ""
	for row in csv.reader(test_file):
		row = [s.replace("NR","0") for s in row]
		for i in range(2,len(row),1):
			row[i] = float(row[i])
		if last == row[0]:
			id_data.append(row)
		else :
			if id_data != [] :
				test_data.append(id_data)
			id_data = []
			id_data.append(row)
			last = row[0]
	test_file.close()
	test_data.append(id_data)
	return test_data

#Learning(Only PM2.5)
#Return : bool (success)
#loop var : p for predict, i for param
def Learning(learning_data):
	global b
	global w

	b_grad = dict((k,0.0) for k in data_item)
	w_grad = dict((k,np.zeros(9,)) for k in data_item)
	b_lr = dict((k,0.0) for k in data_item)
	w_lr = dict((k,np.zeros(9,)) for k in data_item)
	pred = 0.0
	loss_sum = 0.0
	for day_data in learning_data[1:]:
		for p in range(13,27):
			pred = b["PM2.5"]
			for i in range(9,0,-1):
				pred = pred + w["PM2.5"][9-i]*day_data[9][p-i]
			pred = max(0,pred)
			# print ('date:'+str(day_data[9][0]))
			# print ('time:'+str(p))
			# print ('real:'+str(day_data[9][p]))
			# print ('pred:'+str(pred))
			# print ('b:'+str(b["PM2.5"]))
			# print ('w:'+str(w["PM2.5"]))
			loss_sum = loss_sum + (day_data[9][p] - pred)**2
			for i in range(9,0,-1):
				w_grad["PM2.5"][9-i] = w_grad["PM2.5"][9-i] - 2*(day_data[9][p] - pred)*day_data[9][p-i]
				w_lr["PM2.5"][9-i] = w_lr["PM2.5"][9-i] + w_grad["PM2.5"][9-i]**2
				w["PM2.5"][9-i] = w["PM2.5"][9-i] - learn_rate/np.sqrt(w_lr["PM2.5"][9-i]) * w_grad["PM2.5"][9-i]
			b_grad["PM2.5"] = b_grad["PM2.5"] - 2*(day_data[9][p] - pred)
			b_lr["PM2.5"] = b_lr["PM2.5"] + b_grad["PM2.5"]**2
			b["PM2.5"] = b["PM2.5"] - learn_rate/np.sqrt(b_lr["PM2.5"]) * b_grad["PM2.5"] 
	return loss_sum/(240*14)

#predict
#Return : (Output(.csv))
def Predict(test_data):
	global b
	global w

	result_file = open(sys.argv[3],'w')
	wf = csv.writer(result_file)
	wf.writerow(['id','value'])
	pred = 0.0
	for id_data in test_data:
		pred = b["PM2.5"]
		for i in range(2,11):
			pred = pred + w["PM2.5"][i-2]*id_data[9][i]
		wf.writerow([id_data[0][0],max(pred,0)])
	result_file.close()
	return True

#main
train_data = Train_Input()
test_data = Test_Input()
if model_file=="0":
	Init()
else :
	Import_model()
Export_model("model_")
last_loss = 0
print dist
while loss > 10 and (loss - last_loss > 0 or last_loss - loss > dist) :
	if loss > last_loss:
		learn_rate = learn_rate/2
	last_loss = loss
	loss = Learning(train_data)
	# print (last_loss,loss)
	# time.sleep(1)
print loss
Predict(test_data)
Export_model("resultparam_")
print ('time:' + str(time.clock()-begin))