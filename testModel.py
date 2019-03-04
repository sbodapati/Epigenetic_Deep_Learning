import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading binned openness data")
testBinnedOpennessReshaped = np.load("data/pairedData/human/testBinnedOpennessReshaped.npy")
testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201)) # original shape
testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000*201))
testBinnedOpennessReshaped_mean = (testBinnedOpennessReshaped - np.mean(testBinnedOpennessReshaped, axis=1)[:, np.newaxis])
testBinnedOpennessReshaped = testBinnedOpennessReshaped_mean/(np.std(testBinnedOpennessReshaped, axis=1)[:, np.newaxis] + 10**-8)
testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201))
testBinnedOpennessReshaped = np.swapaxes(testBinnedOpennessReshaped, 1, 2)
testBinnedOpennessReshaped.shape # should be (200, 201, 2000)
X = np.reshape(testBinnedOpennessReshaped, (200*201, 2000))

print("Loading gene expression data")
Y = np.genfromtxt("data/pairedData/human/testGeneExpression.txt", delimiter = '\t')


#Linear Regression Baseline
epsilon = 10 ** -8
Y = np.log(Y + epsilon)
Y_mean = Y - np.mean(Y, axis=1)[:,np.newaxis]
Y = Y_mean / (np.std(Y_mean, axis=1)[:,np.newaxis] + 10**-8)
Y = np.reshape(Y, (200*201, 1))
input_size = 2000 
output_size = 1

def get_training_data(X_path="data/pairedData/human/testBinnedOpennessReshaped.npy",
						Y_path="data/pairedData/human/testGeneExpression.txt"):
	"""Loads openness and expression training data for the first 201 samples

	Args:
		X_path (str): path to binned openness data
		Y_path (str): path to gene expression data
	"""
	print("Loading binned openness data")
	# load binned openness data
	testBinnedOpennessReshaped = np.load(X_path)
	testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201)) # original shape
	testBinnedOpennessReshaped = np.swapaxes(testBinnedOpennessReshaped, 1, 2)
	# testBinnedOpennessReshaped.shape should be (200, 201, 2000)
	# 200 genes * 201 samples = 40200 inputs * 2000 1kb bins
	X = np.reshape(testBinnedOpennessReshaped, (200*201, 2000))
	print("Loading gene expression data")
	Y = np.genfromtxt(Y_path, delimiter = '\t')
	epsilon = 10 ** -8
	# use log scale for gene expression
	Y = np.log(Y + epsilon)
	Y = np.reshape(Y, (200*201, 1))
	return X, Y


class Model(nn.Module):
	""" 4-layer neural network """

	def __init__(self, input_size, inputLayerArray):
		"""Initializes model variables and layer weights in the model

		Args:
			input_size (int): size of input
			hidden_size (int): number of hidden units in hidden layers
			output_size (int): size of output
		"""
		super(Model, self).__init__()

		self.model_layers = nn.ModuleList()

		for i in range(len(inputLayerArray)):
			self.model_layers.append(nn.Linear(input_size, inputLayerArray[i]))
			torch.nn.init.xavier_uniform_(self.model_layers[i].weight)
			input_size = inputLayerArray[i]

		self.model_activations = [nn.ReLU() for i in range(len(self.model_layers)-1)]


	def run_all_forward(self, X):
		"""Runs input forward through all layers

		Args:
			X (torch.Tensor): input to model
		Returns:
			out (torch.Tensor): output from model
		"""
		out = X
		for i in range(len(self.model_layers)-1):
			out = self.forward(out, self.model_layers[i], self.model_activations[i])
		out = self.model_layers[-1](out)
		return out
		
	def forward(self, X, layer, activation):
		out = layer(X)
		return activation(out)



def run_training(X, Y, num_epochs = 50):
	"""Runs training on X for specified number of epochs using an Adam
	optimizer and MSE loss. 

	Args:
		X (torch.Tensor): input tensor
		Y (torch.Tensor): target tensor
		num_epochs (int): number of training epochs
	"""
	input_size = 2000 
	inputLayerArray = [1000,100,1] #defines what the input of each layer should be. Make sure the last element is 1.
	print("Loading model")
	model = Model(input_size=input_size, inputLayerArray= inputLayerArray)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0, amsgrad=False)
	loss = nn.MSELoss()
	error_array = np.zeros(num_epochs)

	print("Beginning training")
	for epoch in range(num_epochs):
		optimizer.zero_grad()
		Y_hat = model.run_all_forward(X)
		error = loss(Y_hat, Y)
		error.backward()
		optimizer.step()
		print("Epoch %d: Loss is %f" % (epoch, error))
		error_array[epoch] = error

	np.savetxt("errorVsEpoch.csv", error_array, delimiter=",")
	torch.save(model.state_dict(), 'last_trained_model')

	# below is code to load.
	# the_model = TheModelClass(*args, **kwargs)
	# the_model.load_state_dict(torch.load(PATH))


def main():
	
	X, Y = get_training_data()
	run_training(X=torch.from_numpy(X).float(), Y=torch.from_numpy(Y).float())


if __name__ == "__main__":
	main()
