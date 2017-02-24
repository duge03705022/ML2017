import numpy as np
import sys

filenameA = sys.argv[1];
filenameB = sys.argv[2];

finA = open(filenameA,'r');
finB = open(filenameB,'r');

mA = np.loadtxt(finA,delimiter=',',dtype='int');
mB = np.loadtxt(finB,delimiter=',',dtype='int');

fout = open('ans_one.txt','w');

mC = sorted(mA.dot(mB));

for x in mC:
	fout.write(str(x)+'\n');

