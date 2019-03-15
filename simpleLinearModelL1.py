from __future__ import print_function
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

lamb = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
print("lamb = " + str(lamb), file = sys.stderr)
ff_name = "L1Reg_Training_lambda" + str(lamb) + "out.txt"
with open(ff_name, "w") as ff:
  print("opening FullModel_Training_out.txt", file = ff)

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
                # initialize layers
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

    def __init__(self, type, batch_size = 1000):
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
        # with open(ff_name, "a") as ff:
        #     print('#################################', file = ff)
        #     print('file index: %d'%self.file_index, file = ff)
        #     print('train index: %d'%self.train_index, file = ff)

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

class L1_Model(nn.Module):
    """ N-layer neural network """

    def __init__(self, input_size):
        """Initializes model variables and layer weights in the model

        Args:
            input_size (int): size of input
            inputLayerArray (list of ints): each element specifies the number of neurons in a given layer.
        """
        super(L1_Model, self).__init__()
        self.hidden = nn.Linear(input_size, 1)
        nn.init.xavier_uniform_(self.hidden.weight)
    def forward(self, X):
        """Runs input forward through all layers

        Args:
            X (torch.Tensor): input to model
        Returns:
            out (torch.Tensor): output from model
        """
        return self.hidden(X)
    def getWeight(self):
        print(self.hidden.weight)
        return()

def run_L1_training(num_epochs = 200, batch_size=10000, lamb = 0.0):
    """Runs training specified number of epochs using an Adam
    optimizer and MSE loss.

    Args:
        X (torch.Tensor): input tensor
        Y (torch.Tensor): target tensor
        num_epochs (int): number of training epochs
    """
    input_size = 2000

    #defines what the input of each layer should be. Make sure the last element is 1.
    inputLayerArray = [1000,1000,1000,1]

    with open(ff_name, "a") as ff:
          print("Loading model", file = ff)
    train_data_loader = DataLoader(type='train', batch_size=batch_size)
    dev_data_loader = DataLoader(type='dev')
    model = Model(input_size=input_size, inputLayerArray= inputLayerArray)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0)
    mse_loss = nn.MSELoss()
    train_error_array = np.zeros(num_epochs)
    dev_error_array = np.zeros(num_epochs)
    param_sum_array = np.zeros(num_epochs)

    lastEpoch = train_data_loader.GetEpoch()
    while train_data_loader.GetEpoch() < num_epochs:
        optimizer.zero_grad()
        X,Y = train_data_loader.Next()
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        Y_hat = model.run_all_forward(X)
        train_loss = mse_loss(Y_hat, Y)
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss = reg_loss + torch.sum(torch.abs(param))
        train_loss = train_loss + lamb*reg_loss
        train_loss.backward()
        optimizer.step()
        if train_data_loader.GetEpoch() != lastEpoch:
            error = mse_loss(Y_hat, Y)
            with open(ff_name, "a") as ff:
                print("Epoch %d: Loss is %f" % (train_data_loader.GetEpoch(), error), file = ff)
            train_error_array[train_data_loader.GetEpoch() - 1] = error
            lastEpoch = train_data_loader.GetEpoch()
            np.savetxt("errorVsEpochFullDatasetTrain_lambda" + str(lamb) + ".txt", train_error_array, delimiter=",")
            torch.save(model.state_dict(), 'last_trained_model_onFullDataset_lambda' + str(lamb))
            Xdev, Ydev = dev_data_loader.Next()
            Xdev = torch.from_numpy(Xdev).float()
            Ydev = torch.from_numpy(Ydev).float()
            Ydev_hat = model.run_all_forward(Xdev)
            dev_error = mse_loss(Ydev_hat, Ydev)
            dev_error_array[train_data_loader.GetEpoch() - 1] = dev_error
            np.savetxt("errorVsEpochFullDatasetDev_lambda" + str(lamb) + ".txt", dev_error_array, delimiter=",")
            paramsum = 0.0
            for param in model.parameters():
                paramsum = paramsum + torch.sum(torch.abs(param))
            param_sum_array[train_data_loader.GetEpoch() - 1] = paramsum
            np.savetxt("sumParams_lambda" + str(lamb) + ",txt", param_sum_array, delimiter = ",") 

    torch.save(model.state_dict(), "last_trained_model_onFullDataset_lambda" + str(lamb))

    # below is code to load.
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))

def main():
    with open(ff_name, "a") as ff:
        print('starting', file = ff)
    # run_training()
    run_L1_training(lamb = lamb)


if __name__ == "__main__":
    main()
