import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

data_path = "data/face/"
face_list = ["A","B","C","D","E","F","G","H","I","J"]
index_list = ["00","01","02","03","04","05","06","07","08","09"]
data = []
for face in face_list:
	facedata = []
	for num in index_list:
		faceimg = Image.open(data_path+str(face)+str(num)+".bmp")
		flatdata = np.array(faceimg).flatten()
		facedata.append(flatdata)
		data.append(flatdata)
	facedata = np.array(facedata)
	face_mean = facedata.mean(axis=0)
	

	plt.imshow(face_mean.reshape(64,64), cmap='gray')
	if not os.path.exists("pca/average_img"):
		os.makedirs("pca/average_img")
	plt.savefig("pca/average_img/"+str(face)+".jpg")

data = np.array(data)
average_face = data.mean(axis=0)

plt.imshow(average_face.reshape(64,64), cmap='gray')
if not os.path.exists("pca/average_img"):
	os.makedirs("pca/average_img")
plt.savefig("pca/average_img/average.jpg")

data_ctr = data - data.mean(axis=0,keepdims=True)
u, s, v = np.linalg.svd(data_ctr,full_matrices=False)

fig = plt.figure(figsize=(10, 3))
for i in range(0,9):
	eigenface = v[i].reshape(64,64)

	ax = fig.add_subplot(3, 3, i+1)
	ax.imshow(eigenface, cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()

if not os.path.exists("pca/eigenface_img"):
	os.makedirs("pca/eigenface_img")    
fig.suptitle("Eigenfaces")
fig.savefig("pca/eigenface_img/total.png")
