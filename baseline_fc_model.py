import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt



class Dataset(object):
	"""
	"""
	def __init__(self):
		pass


	def get_batch(batch_size):
		pass


class Model(nn.Module):
	"""
	"""
	def __init__(self, input_size, hidden_size, output_size, weight_range):
		super(Model, self).init()
		self.linear_layers = nn.ModuleList(
				[nn.Linear(input_size, hidden_size),
				 nn.Linear(hidden_size, output_size)])
		self.nonlinearities = [self.get_nonlinearity(hidden_nonlinearity),
							   self.get_nonlinearity("softmax")] # check pytorch docs
		for i, layer in enumerate(self.linear_layers):
			self.init_weights(i, layer, weight_range)


	def init_weights(self, i, layer, weight_range):
		# inplace update of layer weights and biases
		layer.weight.data.uniform_(weight_range[0], weight_range[1])
		layer.bias.data.uniform_(weight_range[0], weight_range[1])


	def get_nonlinearity(self, nonlinearity_name):
		if nonlinearity_name == "relu":
			return nn.ReLU()
		elif nonlinearity_name == "tanh":
			return nn.Tanh()
		elif nonlinearity_name == "sigmoid":
			return nn.Sigmoid()
		elif nonlinearity_name == "softmax":
			return nn.Softmax()

	def forward(self, X):
		output = X
		for layer, nonlinearity in zip(self.linear_layers, self.nonlinearities):
			output = layer(X)
			output = nonlinearity(output)

