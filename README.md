# GenefyHMM
A tool to predict probability of genetic diseases from gene expressions using HMMs with Vector Quantization

## Requirements
Python 2.7 with the folllowing packages installed:
* <b>Numpy</b>
* <b>Scipy</b>
* <b>pickle</b>

## Important Scripts
* <b>myhmm_scaled.py</b>: An implementation of HMM with scaling by @ananthpn
* <b>train.py</b>: Uses the files in the traning folder to train a set of HMMs using the Vector Quantized version of input that is generated.
* <b>test.py</b>: Uses the trained HMMs to emit probabilities for a sequence of real valued inputs through the input.txt file

## Usage
1. Populate the training folder as needed, provide path to it in the function call in train.py if required.
2. Run the script: <b><i> python train.py </i></b>
3. A set of .pkl files should have gotten generated by the previous command
4. Fill a number of real values (comma separated) into test_input.txt.
5. Run the predictor script: <b><i> python test.py </i></b>
