import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import time

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
	def __init__(self, input_size, hidden_size, output_size, init_weight_range):
		super(Model, self).init()
		self.linear_layers = nn.ModuleList(
				[nn.Linear(input_size, hidden_size),
				 nn.Linear(hidden_size, output_size)])
		self.nonlinearities = [self.get_nonlinearity(hidden_nonlinearity),
							   self.get_nonlinearity("softmax")] # check pytorch docs
		for i, layer in enumerate(self.linear_layers):
			self.init_weights(i, layer, init_weight_range)


	def init_weights(self, i, layer, init_weight_range):
		# inplace update of layer weights and biases
		layer.weight.data.uniform_(init_weight_range[0], init_weight_range[1])
		layer.bias.data.uniform_(init_weight_range[0], init_weight_range[1])


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
		return output



class Trainer(object):
	
	def __init__(self,
				 dataset,
				 model,
				 name="",
				 mini_batch_size,
				 num_epochs,
				 learning_rate,
				 momentum,
				 save_freq,
				 test_freq,
				 checkpoint_freq,
				 save_dir):
		self.dataset = dataset
		self.model = model
		self.name = name
		self.mini_batch_size = mini_batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.momentum = momentum

		## TODO: insert loss function here (using L2)
		self.loss_function = lambda y, y_hat: torch.sum((y - y_hat) ** 2)
		self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=self.momentum)
		self.save_freq = save_freq
		self.test_freq = test_freq
		self.checkpoint_freq = checkpoint_freq
		self.save_dir = save_dir
		self.setup()

	def setup():
		if self.save_dir is None:
			self.save_dir = os.path.join(get_logdir())
		self.checkpoints_dir = os.path.join(self.save_dir, "checkpoint_files_{}".format(self.name))
		[utils.makedir(path) for path in [self.save_dir, self.checkpoints_dir]]
		self.checkpoint_file = os.path.join(self.checkpoints_dir, "checkpoint_{}.pth".format(self.name))

		# TODO: add checkpoint loading
		# if os.path.exists(self.checkpoint_file):
		# 	print("Loading from checkpoint: {}".format(self.checkpoint_file))
		# 	checkpoint = torch.load(self.checkpoint_file)
		# 	self.start_epoch = t
		self.start_epoch = 0
		self.loss_data = {"epochs": [], "train_losses": []}

	def save_checkpoint(self, epoch):
	    checkpoint = {"epoch": epoch,
	                  "model_state": self.model.state_dict(),
	                  "optimizer_state": self.optimizer.state_dict()}
	    torch.save(checkpoint, self.checkpoint_file)


	def train(self):
		for epoch in range(self.start_epoch, self.num_epochs):
			train_batches = self.dataset.get_batch(self.mini_batch_size)
			loss_per_batch = []
			for batch in train_batches:
				X, Y = batch[0], batch[1]
				# zero the param gradients
				self.optimizer.zero_grad()
				# get y_hat
				Y_hat = self.model.forward(X)
				loss = self.loss_function(Y_hat, Y)
				loss_per_batch.append(loss)
				self.loss_data["epochs"].append(epoch)
            	self.loss_data["train_losses"].append(float(epoch_loss.data.numpy()))



def run_training(hidden_nonlinearity,
				init_weight_range,
				num_epochs,
				learning_rate,
				momentum,
				mini_batch_size):
	start_time = time.time()
	model = Model(input_size=200,
				  hidden_size=hidden_size,
				  output_size=200,
				  mini_batch_size=mini_batch_size,
				  learning_rate=learning_rate,
				  momentum=momentum) # TODO: change input/output size params
	trainer.train()
	end_time = time.time()
	print("Training time: ", end_time - start_time)




def main():
	# num_runs = 1

	# params
	num_epochs = 100
	learning_rate = 0.2
	momentum = 0.9
	init_weight_range = [-1, 1]
	mini_batch_size = 10
	hidden_layer_size = 4
	hidden_nonlinearity = "relu"

	# results_from_runs = train

	results = run_training(hidden_size,
						   hidden_nonlinearity,
						   init_weight_range,
						   num_epochs,
						   learning_rate,
						   momentum,
						   mini_batch_size)

main()


