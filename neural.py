import numpy as np
import pandas as pd
import torch


def tensor(processed_data, sequence_data_x, sequence_data_y):
    x_np = torch.from_numpy(sequence_data_x)
    y_np = torch.from_numpy(sequence_data_y)
