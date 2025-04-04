# import the neural network
import torch.nn as nn


#https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class CryptoPreedictor(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(CryptoPreedictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # transforms into predictions
    def forward(self, x):
        # only x (ignores y)
        lstm_out, _ = self.lstm(x)
        # use last time-step's output for prediction
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out
