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
import os, pickle, math
from myhmm_scaled import MyHmmScaled as HMM

def testHMMs(inp_file="test_input.txt"):
	inp_vals = map(float, open("test_input.txt","r").read().strip().split(','))
	HMMs = pickle.load(open("trained_HMMs_saved.pkl","rb"))
	codebooks = pickle.load(open("all_codebooks.pkl","rb"))
	vs = pickle.load(open("size_mapping.pkl","rb"))
	
	results = {}
	
	for size in vs.keys():
		# organize input data into vectors of particular size
		c = 0
		vecs = []
		for i in range(len(inp_vals)/int(size)):
			vecs.append(inp_vals[c:c+size])
			c += size
		#print vecs
		# Vector Quantizing
		n = len(vecs)
		vq_seq = map(str,sp.vq(np.reshape(vecs,(n,size)), codebooks[size])[0])
		if len(vq_seq) > 0:
			diseases = vs[size]
			for disease in diseases:
				HMM_obj = HMM("initial.json")
				HMM_obj.A = HMMs[disease]["A"]
				HMM_obj.B = HMMs[disease]["B"]
				HMM_obj.pi = HMMs[disease]["pi"]
				prob = HMM_obj.forward_scaled(vq_seq)
				results[disease] = math.exp(prob)
	
	for i in HMMs.keys():
		if not results.has_key(i):
			results[i] = "Not Enough Data to Predict"
	
	print "RESULTS:"		
	for dis in results.keys():
		print dis.capitalize()+": "+str(results[dis])
		
if __name__=="__main__":
	testHMMs()
