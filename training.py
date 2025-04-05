import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predictor import CryptoPredictor


def tensor(processed_data, sequence_data_x, sequence_data_y):

    # set our input data as tensor (type float)
    x_tensor = torch.from_numpy(sequence_data_x).float()
    # print("X TENSOR: ", x_tensor)
    # reshape it otherwise it causes warning
    y_tensor = torch.from_numpy(sequence_data_y.reshape(-1, 1)).float()
    # print("Y TENSOR: ", y_tensor)


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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # create our model
    model = CryptoPredictor(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(num_epochs):
        for batch_x, batch_y in data_loader:
            # forward pass: model predictions
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # backwards pass: optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate model after training
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(x_tensor)
    # print("Predictions Normalized:", predicted_normalized)

    # use the original statistics from processed_data (not from the normalized dataframe)
    close_mean = processed_data["close_mean"]
    close_std = processed_data["close_std"]

    # converted it back to the original scale
    predicted_original = predicted_normalized * close_std + close_mean

    # print("Predictions Original Scale:", predicted_original)

    return predicted_original, y_tensor

    # evaluate the predictions
