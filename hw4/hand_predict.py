import numpy as np
import csv
from sklearn.svm import LinearSVR as SVR
from PIL import Image
from hand_gen import get_eigenvalues

# Train a linear SVR

npzfile = np.load('hand_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=1)
svr.fit(X, y)

# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

# predict
# testdata = np.load('data.npz')

testdata = []
ratio = 0.05
for i in range(1,482):
	img = Image.open("data/hand/hand.seq%s.png" % str(i))
	width = int(img.size[0]*ratio)
	height = int(img.size[1]*ratio)
	img = img.resize( (width, height), Image.BILINEAR )
	img_data = np.array(img).flatten()
	testdata.append(img_data)
testdata = np.array(testdata).reshape(481,width*height)

test_X = get_eigenvalues(testdata)
# for i in range(200):
#     data = testdata[str(i)]
#     vs = get_eigenvalues(data)
#     test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)

for i, d in enumerate(pred_y):
    print(i,d)
