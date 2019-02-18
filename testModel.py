import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading binned openness data")
testBinnedOpennessReshaped = np.load("data/pairedData/human/testBinnedOpennessReshaped.npy")
testBinnedOpennessReshaped = np.reshape(testBinnedOpennessReshaped, (200, 2000, 201)) # original shape
testBinnedOpennessReshaped = np.swapaxes(testBinnedOpennessReshaped, 1, 2)
testBinnedOpennessReshaped.shape # should be (200, 201, 2000)
X = np.reshape(testBinnedOpennessReshaped, (200*201, 2000))

print("Loading gene expression data")
Y = np.genfromtxt("data/pairedData/human/testGeneExpression.txt", delimiter = '\t')
epsilon = 10 ** -8
Y = np.log(Y + epsilon)
Y = np.reshape(Y, (200*201, 1))
input_size = 2000 
output_size = 1

class Model(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Model, self).__init__()
		self.input = nn.Linear(input_size, hidden_size)
		self.hidden1 = nn.Linear(hidden_size, hidden_size)
		self.hidden2 = nn.Linear(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)

		torch.nn.init.xavier_uniform_(self.input.weight)
		torch.nn.init.xavier_uniform_(self.hidden1.weight)
		torch.nn.init.xavier_uniform_(self.hidden2.weight)
		torch.nn.init.xavier_uniform_(self.out.weight)

		# self.hidden_layers = [nn.Linear(hidden_size, hidden_size) for i in num_hidden_layers]
		self.layers = nn.ModuleList([self.input, self.hidden1, self.hidden2, self.out])
		self.activations = [nn.ReLU() for i in range(len(self.layers))]


	def run_all_forward(self, X):
		out = X
		for i in range(len(self.layers)-1):
			out = self.forward(out, self.layers[i], self.activations[i])

		out = self.out(out)
		# print(out)
		return out
		
	def forward(self, X, layer, activation):
		out = layer(X)
		return activation(out)




def run_training(X, Y, num_epochs = 5):
	model = Model(input_size=input_size, hidden_size=2000, output_size=output_size)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0, amsgrad=False)
	loss = nn.MSELoss()
	error_array = np.zeros(num_epochs)

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


print("Beginning training")

run_training(torch.from_numpy(X).float(), torch.from_numpy(Y).float())



