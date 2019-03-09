import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# print("Loading binned openness data")
# testBinnedOpennessReshaped = np.load("data/pairedData/human/testBinnedOpennessReshaped.npy")
# testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201)) # original shape
# testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000*201))
# testBinnedOpennessReshaped_mean = (testBinnedOpennessReshaped - np.mean(testBinnedOpennessReshaped, axis=1)[:, np.newaxis])
# testBinnedOpennessReshaped = testBinnedOpennessReshaped_mean/(np.std(testBinnedOpennessReshaped, axis=1)[:, np.newaxis] + 10**-8)
# testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201))
# testBinnedOpennessReshaped = np.swapaxes(testBinnedOpennessReshaped, 1, 2)
# testBinnedOpennessReshaped.shape # should be (200, 201, 2000)
# X = np.reshape(testBinnedOpennessReshaped, (200*201, 2000))
#
# print("Loading gene expression data")
# Y = np.genfromtxt("data/pairedData/human/testGeneExpression.txt", delimiter = '\t')


# #Linear Regression Baseline
# epsilon = 10 ** -8
# Y = np.log(Y + epsilon)
# Y_mean = Y - np.mean(Y, axis=1)[:,np.newaxis]
# Y = Y_mean / (np.std(Y_mean, axis=1)[:,np.newaxis] + 10**-8)
# Y = np.reshape(Y, (200*201, 1))
# input_size = 2000
# output_size = 1

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

	################################################
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
	epsilon = 10 ** -8
	Y = np.log(Y + epsilon)
	Y_mean = Y - np.mean(Y, axis=1)[:,np.newaxis]
	Y = Y_mean / (np.std(Y_mean, axis=1)[:,np.newaxis] + 10**-8)
	Y = np.reshape(Y, (200*201, 1))

	return X, Y


class Model(nn.Module):
	""" N-layer neural network """

	def __init__(self, input_size, inputLayerArray):
		"""Initializes model variables and layer weights in the model

		Args:
			input_size (int): size of input
			inputLayerArray (list of ints): each element specifies the number of neurons in a given layer.
		"""
		super(Model, self).__init__()

		self.model_layers = nn.ModuleList()

		for i in range(len(inputLayerArray)):
			self.model_layers.append(nn.Linear(input_size, inputLayerArray[i]))
			nn.init.xavier_uniform_(self.model_layers[i].weight)
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

class DataLoader():
	X_path = '/oak/stanford/groups/slqi/tdaley/pairedData/Epigenetic_Deep_Learning/data/splits/Merged/logNormalizedBinnedOpennessReshaped'
	Y_path = '/oak/stanford/groups/slqi/tdaley/pairedData/Epigenetic_Deep_Learning/data/splits/Merged/logNormalizedGeneExpressionReshaped'
	dev = None
	test = None
	train = None
	train_index = None
	file_index = None
	fileEnd = None
	currentNumpyArray_X = None
	batch_size = None
	epoch_num = 0

	def __init__(self, type, batch_size):
		if type == 'dev':
			self.dev = True
		if type == 'test':
			self.test = True
		if type == 'train':
			self.train = True
			self.currentNumpyArray_X = np.load(self.X_path + 'Train_0.npy')
			self.currentNumpyArray_Y = np.load(self.Y_path + 'Train_0.npy')
			self.batch_size = batch_size
			self.train_index = 0
			self.file_index = 0
			self.fileEnd = False

		return

	def Next(self):
		print('#################################')
		print('file index: %d'%self.file_index)
		print('train index: %d'%self.train_index)

		if self.dev:
			data_X = np.load(self.X_path + 'Dev.npy')
			data_Y = np.load(self.Y_path + 'Dev.npy')
			self.epoch_num = self.epoch_num + 1
			return data_X, data_Y
		if self.test:
			data_X = np.load(self.X_path + 'Test.npy')
			data_Y = np.load(self.Y_path + 'Dev.npy')
			self.epoch_num = self.epoch_num + 1
			return data_X, data_Y

		###########
		#code only applies to training sets#
		###########
		if self.fileEnd: #at the end of the file
			if self.file_index == 99:
				self.file_index = 0
				self.epoch_num = self.epoch_num + 1

			else:
				self.file_index = self.file_index + 1

			self.currentNumpyArray_X = np.load(self.X_path + 'Train_%d.npy' % self.file_index)
			self.currentNumpyArray_Y = np.load(self.Y_path + 'Train_%d.npy' % self.file_index)
			self.fileEnd = False
			self.train_index = 0

		lenOfCurrentArray = self.currentNumpyArray_X.shape[0]

		if self.train_index + self.batch_size > lenOfCurrentArray:
			dataToReturn_X = self.currentNumpyArray_X[self.train_index:, :]
			dataToReturn_Y = self.currentNumpyArray_Y[self.train_index:, :]
			self.fileEnd = True
			return dataToReturn_X, dataToReturn_Y
		else:
			dataToReturn_X = self.currentNumpyArray_X[self.train_index:self.train_index + self.batch_size, :]
			dataToReturn_Y = self.currentNumpyArray_Y[self.train_index:self.train_index + self.batch_size, :]
			self.train_index = self.train_index + self.batch_size
			return dataToReturn_X, dataToReturn_Y

	def GetEpoch(self):
		return self.epoch_num






def run_training(num_epochs = 5000, batch_size=20000):
	"""Runs training specified number of epochs using an Adam
	optimizer and MSE loss. 

	Args:
		X (torch.Tensor): input tensor
		Y (torch.Tensor): target tensor
		num_epochs (int): number of training epochs
	"""

	# X, Y = get_training_data()
	# X = torch.from_numpy(X).float()
	# Y = torch.from_numpy(Y).float()

	input_size = 2000

	#defines what the input of each layer should be. Make sure the last element is 1.
	inputLayerArray = [1000,1000,1]

	print("Loading model")
	data_loader = DataLoader(type='train', batch_size=batch_size)
	model = Model(input_size=input_size, inputLayerArray= inputLayerArray)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0)
	loss = nn.MSELoss()
	error_array = np.zeros(num_epochs)

	print("Beginning training")
	lastEpoch = data_loader.GetEpoch()
	while data_loader.GetEpoch()<num_epochs:
		optimizer.zero_grad()
		X,Y = data_loader.Next()
		X = torch.from_numpy(X).float()
		Y = torch.from_numpy(Y).float()
		Y_hat = model.run_all_forward(X)
		error = loss(Y_hat, Y)
		error.backward()
		optimizer.step()
		if data_loader.GetEpoch() != lastEpoch:
			print("Epoch %d: Loss is %f" % (data_loader.GetEpoch(), error))
			error_array[data_loader.GetEpoch()] = error
			lastEpoch = data_loader.GetEpoch()

	np.savetxt("errorVsEpoch_onFullDataset.csv", error_array, delimiter=",")
	torch.save(model.state_dict(), 'last_trained_model_onFullDataset')

	# below is code to load.
	# the_model = TheModelClass(*args, **kwargs)
	# the_model.load_state_dict(torch.load(PATH))


def main():
	
	run_training()


if __name__ == "__main__":
	main()
