import torch


def tensor(processed_data, sequence_data_x, sequence_data_y):

    # set our input data as tensor
    x_np = torch.from_numpy(sequence_data_x)
    print("X TENSOR: ", x_np)
    y_np = torch.from_numpy(sequence_data_y)
    print("Y TENSOR: ", y_np)

    # build our initial model

