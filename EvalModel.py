import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys




ff_name = "./EvalResults/Evaluating_Test_with_unreg.txt"


with open(ff_name, "w") as ff:
    print("opening Evaluating_Test_with_unreg.txt", file = ff)

#defines what the input of each layer should be. Make sure the last element is 1.
inputLayerArray = [1000,1000,1000,1]
input_size = 2000

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
    epoch_num = 1
    test_size = None

    def __init__(self, type, batch_size=10000):
        if type == 'dev':
            self.dev = True
        if type == 'test':
            self.test = True
            self.currentNumpyArray_X = np.load(self.X_path + 'Test.npy')
            self.currentNumpyArray_Y = np.load(self.Y_path + 'Test.npy')
            self.train_index = 0
            self.fileEnd = False
            self.test_size = self.currentNumpyArray_X.shape[0]

        if type == 'train': #will not be using this for this model
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
        # 	print('#################################', file = ff)
        # 	print('file index: %d'%self.file_index, file = ff)
        # 	print('train index: %d'%self.train_index, file = ff)

        if self.dev:
            data_X = np.load(self.X_path + 'Dev.npy')
            data_Y = np.load(self.Y_path + 'Dev.npy')
            self.epoch_num = self.epoch_num + 1
            return data_X, data_Y

        if self.test:
            data_X = self.currentNumpyArray_X[self.train_index,:]
            data_Y = self.currentNumpyArray_Y[self.train_index,:]
            self.train_index = self.train_index + 1
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

    def GetTestSize(self):
        return self.test_size

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


def run_eval(num_epochs = 500, batch_size=10000):
    """Runs training specified number of epochs using an Adam
    optimizer and MSE loss.

    Args:
        X (torch.Tensor): input tensor
        Y (torch.Tensor): target tensor
        num_epochs (int): number of training epochs
    """

    #defines what the input of each layer should be. Make sure the last element is 1.

    with open(ff_name, "a") as ff:
        print("Loading model", file = ff)
    test_data_loader = DataLoader(type='test', batch_size=batch_size)

    model = Model(input_size=input_size, inputLayerArray= inputLayerArray)
    model.load_state_dict(torch.load('/oak/stanford/groups/slqi/bodapati/EpigeneticDeeplearning/trainedModelCheckpoints/FullDataset/FullDatasetCheckpoint_295'))
    model.eval()
    test_size = test_data_loader.GetTestSize()
    exp_array = np.zeros((test_size,2))
    index = 0
    while index < test_size:
        X,Y = test_data_loader.Next()
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        Y_hat = model.run_all_forward(X)


        with open(ff_name, "a") as ff:
            print("Epoch %d: Actual Exp - %f | Predicted Exp - %f" % (index, Y, Y_hat), file = ff)

        exp_array[index, 0] = Y
        exp_array[index, 1] = Y_hat

    np.savetxt("Eval_no_reg.csv", exp_array, delimiter=",")

def main():
    with open(ff_name, "a") as ff:
        print('starting', file = ff)
    run_eval()

if __name__ == "__main__":
    main()
