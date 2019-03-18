import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys



# model_path = "last_trained_model_onFullDataset"

ff_name = "Dev_Error_Full_Training_decreasing_nodes_300epochs_out.txt"
with open(ff_name, "w") as ff:
	print("opening FullModel_Training_out.txt", file = ff)

#defines what the input of each layer should be. Make sure the last element is 1.
inputLayerArray = [1000,500,250,1]
input_size = 2000


class Model(nn.Module):
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

	def __init__(self, type, batch_size=10000):
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


def run_training(num_epochs = 300, batch_size=10000):
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

	#defines what the input of each layer should be. Make sure the last element is 1.
	# inputLayerArray = [1000,1000,1000,1]

	with open(ff_name, "a") as ff:
		print("Loading model", file = ff)
	train_data_loader = DataLoader(type='train', batch_size=batch_size)
	dev_data_loader = DataLoader(type='dev', batch_size=batch_size)
	model = Model(input_size=input_size, inputLayerArray= inputLayerArray)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0)
	loss = nn.MSELoss()
	error_array = np.zeros(num_epochs)
	lastEpoch = train_data_loader.GetEpoch()
	totalbatchMSE = 0
	numofbatches = 0

	while train_data_loader.GetEpoch()<num_epochs:
		optimizer.zero_grad()
		X,Y = train_data_loader.Next()
		X = torch.from_numpy(X).float()
		Y = torch.from_numpy(Y).float()
		Y_hat = model.run_all_forward(X)
		train_error = loss(Y_hat, Y)

		totalbatchMSE = totalbatchMSE + train_error
		numofbatches = numofbatches + 1

		train_error.backward()
		optimizer.step()
		if train_data_loader.GetEpoch() != lastEpoch:
			#run dev MSE
			X_d,Y_d = dev_data_loader.Next()
			X_d = torch.from_numpy(X_d).float()
			Y_d = torch.from_numpy(Y_d).float()
			Y_d_hat = model.run_all_forward(X_d)
			dev_error = loss(Y_d_hat, Y_d)

			totalbatchMSE = totalbatchMSE/numofbatches

			with open(ff_name, "a") as ff:
				print("Epoch %d: Train MSE is %f | Dev MSE is %f" % (train_data_loader.GetEpoch(), totalbatchMSE, dev_error), file = ff)

			totalbatchMSE = 0
			numofbatches = 0

			error_array[train_data_loader.GetEpoch()] = train_error
			lastEpoch = train_data_loader.GetEpoch()
			# np.savetxt("errorVsEpoch_onFullDataset.csv", error_array, delimiter=",")
			torch.save(model.state_dict(), 'last_trained_Full_Model')

		if train_data_loader.GetEpoch()%5 == 0:
			torch.save(model.state_dict(), './trainedModelCheckpoints/FullDataset_changinglayers_Checkpoint_%d'%train_data_loader.GetEpoch())

	np.savetxt("errorVsEpoch_changinglayers_onFullDataset.csv", error_array, delimiter=",")
	torch.save(model.state_dict(), 'last_trained_Full_Model_changinglayers')

	# below is code to load.
	# the_model = TheModelClass(*args, **kwargs)
	# the_model.load_state_dict(torch.load(PATH))


def main():
	with open(ff_name, "a") as ff:
		print('starting', file = ff)
	run_training()


if __name__ == "__main__":
	main()
