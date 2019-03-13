import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys



lambd = float(sys.argv[1])
ff_name = "L1Reg_Training_%f_out.txt"%lambd


with open(ff_name, "w") as ff:
    print("opening L1_Training_out_with_lambda:%f.txt" %lambd, file = ff)

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


def run_L1_training(num_epochs = 300, batch_size=10000):
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
    train_data_loader = DataLoader(type='train', batch_size=batch_size)
    dev_data_loader = DataLoader(type='dev', batch_size=batch_size)
    model = L1_Model(input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0)
    loss = nn.MSELoss()
    error_array = np.zeros(num_epochs)


    lastEpoch = train_data_loader.GetEpoch()
    lastMSE = 100000001
    currentMSE = 10000000

    while train_data_loader.GetEpoch()<num_epochs or abs(lastMSE-currentMSE)<10**-5:
        optimizer.zero_grad()
        X,Y = train_data_loader.Next()
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        Y_hat = model.forward(X)
        train_error = loss(Y_hat, Y)

        all_linear1_params = torch.cat([x.view(-1) for x in model.parameters()])
        l1_reg = torch.norm(all_linear1_params, 1)

        train_error = train_error + l1_reg * lambd

        if train_data_loader.GetEpoch() != lastEpoch:
            #run dev MSE
            X_d,Y_d = dev_data_loader.Next()
            X_d = torch.from_numpy(X_d).float()
            Y_d = torch.from_numpy(Y_d).float()
            Y_d_hat = model.forward(X_d)
            dev_error = loss(Y_d_hat, Y_d)
            dev_error = dev_error + l1_reg * lambd

            lastMSE = currentMSE
            currentMSE = dev_error

            with open(ff_name, "a") as ff:
                print("Epoch %d: Train MSE is %f | Dev MSE is %f" % (train_data_loader.GetEpoch(), train_error, dev_error), file = ff)
            error_array[train_data_loader.GetEpoch()-1] = train_error
            lastEpoch = train_data_loader.GetEpoch()
            torch.save(model.state_dict(), 'last_trained_Lasso_Model_lam:%f'%lambd)

        train_error.backward()
        optimizer.step()


    np.savetxt("Lasso_error_vs_epoch_lam:%f.csv"%lambd, error_array, delimiter=",")
    torch.save(model.state_dict(), 'last_trained_Lasso_Model_lam:%f'%lambd)




def main():
    with open(ff_name, "a") as ff:
        print('starting', file = ff)
    run_L1_training()


if __name__ == "__main__":
    main()
