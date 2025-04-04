import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def tensor(processed_data, sequence_data_x, sequence_data_y):

    # set our input data as tensor (type float)
    x_tensor = torch.from_numpy(sequence_data_x).float()
    print("X TENSOR: ", x_tensor)
    y_tensor = torch.from_numpy(sequence_data_y).float()
    print("Y TENSOR: ", y_tensor)


    # parameters for training
    input_size = 6
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # create our data loader
    #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    dataset = TensorDataset(x_tensor, y_tensor)
