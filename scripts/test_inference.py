import torch
import matplotlib.pyplot as plt
import sys
from torch.utils.data import DataLoader
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import H1Loss, LpLoss
from neuralop.losses.data_losses import MSELoss
from inspect_processed_data import get_numerical_data_and_validity_ndarray
from weather_dataset import NumericalWeatherDataset

device = "cpu"

import numpy as np
import argparse
import os.path

parser = argparse.ArgumentParser("train_numerical_model")
parser.add_argument("train_sample_left_index", help="Left bound (inclusive) of datasets to use for training", type=int)
parser.add_argument("save_checkpoint_directory_path", help="Path to load the stored numerical model", type=str)
args = parser.parse_args()

# create a dataset for the training samples
train_l = args.train_sample_left_index
ckpt_path = args.save_checkpoint_directory_path

model = FNO.from_checkpoint(save_folder=os.path.dirname(ckpt_path), save_name=os.path.basename(ckpt_path))
model = model.to(device=device)
input = torch.from_numpy(get_numerical_data_and_validity_ndarray(train_l, train_l+1))
print(input.shape)

torch.set_printoptions(profile="full")
tensor = model(input)
print(tensor)

# Visualizing predictions
# ------------------------
# Let's take a look at what our model's predicted outputs look like.
# We wll compare the inputs, ground-truth outputs, and model predictions side by side.
#
# Note that in this example, we train on a very small resolution for
# a very small number of epochs. In practice, we would train at a larger
# resolution on many more samples.

# test_samples = test_loaders["one"].dataset

# fig = plt.figure(figsize=(7, 7))
# for index in range(3):
#     data = test_samples[index]
#     # Input
#     x = data["x"]
#     # Ground-truth output
#     y = data["y"]
#     # Model prediction
#     out = model(x.unsqueeze(0))

#     # # Plot input
#     # ax = fig.add_subplot(3, 3, index * 3 + 1)
#     # ax.imshow(x[0], cmap="gray")
#     # if index == 0:
#     #     ax.set_title("Input x")
#     # plt.xticks([], [])
#     # plt.yticks([], [])

#     # Plot ground-truth output
#     # ax = fig.add_subplot(3, 3, index * 3 + 2)
#     # ax.imshow(y.squeeze())
#     # if index == 0:
#     #     ax.set_title("Ground-truth output")
#     # plt.xticks([], [])
#     # plt.yticks([], [])

#     # Plot model prediction
#     # ax = fig.add_subplot(3, 3, index * 3 + 3)
#     # ax.imshow(out.squeeze().detach().numpy())
#     # if index == 0:
#     #     ax.set_title("Model prediction")
#     # plt.xticks([], [])
#     # plt.yticks([], [])

# # fig.suptitle("FNO predictions on 16x16 Darcy-Flow data", y=0.98)
# # plt.tight_layout()
# # fig.show()