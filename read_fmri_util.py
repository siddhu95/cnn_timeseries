import os
import numpy as np
import scipy.io
V = 264
T = 137
def parse_id(name):
	name = os.path.splitext(os.path.basename(name))[0]
	return name.split('_')[0]
def read_data(pathx,pathy,centres):
	centre_map = {'0003':3,'0006':6,'0009':9,'0010':10,'0018':18}

	label = scipy.io.loadmat(os.path.join(pathy,'class_label.mat'))
	label = label['class_label']
	label = np.array((1+label[0])/2,dtype='int')
	
	p_id = []
	l = []
	onlymatfiles = [f for f in os.listdir(pathx) if (os.path.isfile(os.path.join(pathx, f)) and f.endswith('.mat'))]
	
	A = np.empty((0,V,T))
	for i,g in enumerate(onlymatfiles):
		if(centre_map[g[0:4]] in centres):
			D = scipy.io.loadmat(os.path.join(pathx,g))
			A = np.append(A,np.reshape(D['ROI_time_series'],(1,V,T)),axis=0)
			p_id.append(parse_id(g))
			l.append(label[i])
	    

	return (A,l,p_id)

def demean(data):
	data = data - np.mean(data,axis=2,keepdims=1)
	return data

def standardize(data):
	mean = np.mean(data,axis=2,keepdims=1)
	sigma = np.std(data,axis=2,keepdims=1)
	data = np.divide(data-mean,sigma)
	return data

def random_split(data,label,ratio):
	data = np.array(data)
	N = data.shape[0]//4
	no_train = max(1,int(ratio*N))
	ind = np.multiply(np.random.permutation(N),4)
	train_ind = ind[:no_train]
	test_ind = ind[no_train:]


	train_ind = np.array(np.hstack([train_ind,np.add(train_ind,1),np.add(train_ind,2),np.add(train_ind,3)]),dtype='int')
	test_ind = np.array(np.hstack([test_ind,np.add(test_ind,1),np.add(test_ind,2),np.add(test_ind,3)]),dtype='int')
	
	print train_ind.shape
	print 'hi'
	X_train = data[train_ind]
	X_test = data[test_ind]

	label = np.array(label)
	Y_train = label[train_ind]
	Y_test = label[test_ind]
	return (X_train,X_test,Y_train,Y_test)


