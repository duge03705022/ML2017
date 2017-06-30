import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

def print_fig_origin(facedata,name):
	fig = plt.figure(figsize=(13, 10))
	i = 0
	for face in facedata:
		for pic in face:
			i = i+1
			ax = fig.add_subplot(10, 10, i)
			ax.imshow(pic.reshape(64,64), cmap='gray')
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
			plt.tight_layout()

	if not os.path.exists("pca/recover" ):
		os.makedirs("pca/recover" )    
	fig.suptitle(name)
	fig.savefig("pca/recover/%s.png" % name)
	return

def recovery(data,v,num):
	x_reduce = []
	for pic in data:
		eigen = []
		for i in range(0,num):
			x_ctr = np.array(pic - data_mean)[np.newaxis]
			v[i] = np.array(v[i])
			tmpvec = np.dot(x_ctr,v[i])
			eigen.append(tmpvec)
		x_reduce.append(eigen)
	x_reduce = np.array(x_reduce)

	res = []
	for pic in x_reduce:
		rec = data_mean
		for i in range(0,num):
			rec = rec + pic[i]*v[i]
		res.append(rec)
	res = np.array(res)
	return res

data_path = "data/face/"
face_list = ["A","B","C","D","E","F","G","H","I","J"]
index_list = ["00","01","02","03","04","05","06","07","08","09"]
data = []
for face in face_list:
	facedata = []
	for num in index_list:
		faceimg = Image.open(data_path+str(face)+str(num)+".bmp")
		flatdata = np.array(faceimg).flatten()
		data.append(flatdata)

data = np.array(data).reshape(10,10,4096)
print_fig_origin(data,"origin_face")
data = data.reshape(100,4096)
data_mean = data.mean(axis=0)
data_ctr = data - data.mean(axis=0,keepdims=True)
u, s, v = np.linalg.svd(data_ctr,full_matrices=False)

res = recovery(data,v,5)
print_fig_origin(res.reshape(10,10,4096),"recovered_face")

for i in range(5,101):
	res = recovery(data,v,i)
	rmse = np.sqrt(((data - res) ** 2).mean()/2500)*100
	if rmse < 1:
		print(i,rmse)
		print_fig_origin(res.reshape(10,10,4096),"lower0.01_%s" % str(i))
		break