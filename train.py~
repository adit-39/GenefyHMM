'''
	Author: Aditya Dalwani <adit.39@gmail.com>
	Date: 8th January, 2016
	Description:
		Script performs the following:
		1. Takes a path to a folder containing training files with name <disease_name>.soft
		2. Generates a codebook based on the gene expression vectors of all diseases that will be used in vector quantization process
		3. Generates VQ files for each disease
		4. Creates a Discrete HMM for each disease and trains it using the Baum-Welch unsupervised training algorithm
		5. Pickles trained models and the codebook
'''

import scipy.cluster.vq as sp
import numpy as np
import os, pickle
from myhmm_scaled import MyHmmScaled as HMM

def obtain_training_data_old(trng_path="./training"):
	trng_data = {}
	vs = {}
	dis_files = os.listdir(trng_path)
	for df in dis_files:
		disease = df.split(".")[0]
		trng_data[disease] = []
		f = open(trng_path+'/'+df,"r")
		l = f.readlines()
		f.close()
		flag = 0
		for i in range(len(l)):
			if flag == 1:
				flag = 2
				continue
			if l[i].strip() == '!dataset_table_begin':
				flag = 1
			elif l[i].strip() == '!dataset_table_end':
				break
			elif flag == 2:
				trng_data[disease].append(map(float, l[i].strip().split()[2:]))
		d_len = len(trng_data[disease][0])
		if not vs.has_key(d_len):
			vs[d_len]= []
		vs[d_len].append(disease)
	return trng_data, vs

def obtain_training_data(trng_path="./training"):
	trng_data = {}
	vs = {}
	dis_files = os.listdir(trng_path)
	for df in dis_files:
		disease = df.split(".")[0]
		trng_data[disease] = []
		f = open(trng_path+'/'+df,"r")
		l = f.readlines()
		f.close()
		for i in range(len(l)):
			#print l[i][:-2]
			s1 = l[i][:-2]
			s2 = s1.replace("null","0")
			#print s
			trng_data[disease].append(map(float, s2.split(",")[1:]))
		d_len = len(trng_data[disease][0])
		if not vs.has_key(d_len):
			vs[d_len]= []
		vs[d_len].append(disease)
	pickle.dump(vs, open("size_mapping.pkl","wb"))
	return trng_data, vs

def vector_quantize(data_dict, vs, bins):
	codebooks = {}
	vq_data = {}
	for size in vs.keys():
		all_size_data = []
		for disease in vs[size]:
			all_size_data.extend(data_dict[disease])
		#whitened = sp.whiten(all_size_data)
		#codebooks[size] = sp.kmeans(whitened, bins)[0]
		codebooks[size] = sp.kmeans(np.asarray(all_size_data), bins)[0]
	pickle.dump(codebooks,open("all_codebooks.pkl","wb"))
	for dis in data_dict.keys():
		n = len(data_dict[dis])
		m = len(data_dict[dis][0])
		vq_data[dis] = map(str,sp.vq(np.reshape(data_dict[dis],(n,m)), codebooks[len(data_dict[dis][0])])[0])
	return vq_data
	
def train_HMMs(vq_dict, vs):
	HMMs = {}
	HMMs_save = {}
	for disease in vq_dict.keys():
		size = 0
		for s in vs.keys():
			if disease in vs[s]:
				size=s
				break
		HMMs[disease] = HMM("initial.json")
		HMMs[disease].forward_backward_multi_scaled([vq_dict[disease]])
		HMMs_save[disease] = {}
		HMMs_save[disease]["A"] = HMMs[disease].A
		HMMs_save[disease]["B"] = HMMs[disease].B
		HMMs_save[disease]["pi"] = HMMs[disease].pi
		
	pickle.dump(HMMs_save, open("trained_HMMs_saved.pkl","wb"))	


if __name__=="__main__":
	bins = 16
	trng, vec_sizes = obtain_training_data()
	vq_data = vector_quantize(trng, vec_sizes, bins)
	train_HMMs(vq_data, vec_sizes)
