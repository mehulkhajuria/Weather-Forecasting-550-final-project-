import torch
from torch import Tensor
import matplotlib.pyplot as plt
import sys
from torch.utils.data import DataLoader
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import H1Loss, LpLoss
from neuralop.losses.data_losses import MSELoss
from inspect_processed_data import get_rain_data_and_validity_ndarray, get_numerical_data_and_validity_ndarray, provide_normalizer
from weather_dataset import NumericalWeatherDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

device = "cpu"

import numpy as np
import argparse
import os.path

parser = argparse.ArgumentParser("train_numerical_model")
parser.add_argument("train_sample_left_index", help="Left bound (inclusive) of datasets to use for training", type=int)
parser.add_argument("train_sample_right_index", help="Right bound (exclusive) of datasets to use for training", type=int)
parser.add_argument("test_sample_left_index", help="Left bound (inclusive) of datasets to use for testing", type=int)
parser.add_argument("test_sample_right_index", help="Right bound (exclusive) of datasets to use for testing", type=int)
parser.add_argument("save_checkpoint_directory_path", help="Path to save the stored numerical model", type=str)
parser.add_argument("epochs", help="number of epochs to train for", type=int)
args = parser.parse_args()

# create a dataset for the training samples
train_l = args.train_sample_left_index
train_r = args.train_sample_right_index
test_l = args.test_sample_left_index
test_r = args.test_sample_right_index
ckpt_path = args.save_checkpoint_directory_path
epochs = args.epochs

train_numerical_sample_tensor = torch.from_numpy(get_numerical_data_and_validity_ndarray(train_l, train_r)).type(torch.FloatTensor)
train_rain_sample_tensor = torch.from_numpy(get_rain_data_and_validity_ndarray(train_l, train_r)).type(torch.FloatTensor)
print(type(train_numerical_sample_tensor), type(train_rain_sample_tensor))
print(train_numerical_sample_tensor.shape, train_rain_sample_tensor.shape)

train_dataset = NumericalWeatherDataset(train_numerical_sample_tensor, train_rain_sample_tensor)
train_loader = DataLoader(train_dataset)

#create a normalizer for the inputs
#prevent the validity mask from being normalized
# validity_preserving_input_normalize_mask = np.array([np.ones((3 * 5, 32, 32)), np.zeros((3 * 5,32,32))])
# validity_preserving_input_normalize_mask = np.transpose(validity_preserving_input_normalize_mask, (1, 0, 2, 3))
# attr, v, x, y = np.shape(validity_preserving_input_normalize_mask)
# validity_preserving_input_normalize_mask = np.reshape(validity_preserving_input_normalize_mask, (attr * v, x, y))
# validity_preserving_input_normalize_mask = torch.from_numpy(validity_preserving_input_normalize_mask)

train_in_normalizer = provide_normalizer(train_dataset, "x", 3*5*2)

#can't use from_dataset, there is some kind of broadcasting bug that prevent channel variant masks.
# train_in_normalizer = UnitGaussianNormalizer.from_dataset(
#     dim=[0, 2, 3],
#     dataset = train_dataset,
#     mask = validity_preserving_input_normalize_mask,
#     keys=["x"]
# )

#create a normalizer for the outputs
# validity_preserving_output_normalize_mask = np.array([np.ones((2, 32, 32)), np.zeros((2,32,32))])
# validity_preserving_output_normalize_mask = np.transpose(validity_preserving_output_normalize_mask, (1, 0, 2, 3))
# attr, v, x, y = np.shape(validity_preserving_output_normalize_mask)
# validity_preserving_output_normalize_mask = np.reshape(validity_preserving_output_normalize_mask, (attr * v, x, y))
# validity_preserving_output_normalize_mask = torch.from_numpy(validity_preserving_output_normalize_mask)

train_out_normalizer = provide_normalizer(train_dataset, "y", 2*2)
# train_out_normalizer = UnitGaussianNormalizer.from_dataset(
#     dim=[0, 2, 3],
#     dataset = train_dataset,
#     mask = validity_preserving_output_normalize_mask,
#     keys=["y"]
# )


train_processor = DefaultDataProcessor(in_normalizer=train_in_normalizer, out_normalizer=train_out_normalizer)

#do the same but for the testing data

test_numerical_sample_tensor = torch.from_numpy(get_numerical_data_and_validity_ndarray(test_l, test_r)).type(torch.FloatTensor)
test_rain_sample_tensor = torch.from_numpy(get_rain_data_and_validity_ndarray(test_l, test_r)).type(torch.FloatTensor)
print(type(test_numerical_sample_tensor), type(test_rain_sample_tensor))
print(test_numerical_sample_tensor.shape, test_rain_sample_tensor.shape)

test_dataset = NumericalWeatherDataset(test_numerical_sample_tensor, test_rain_sample_tensor)
test_loader = DataLoader(test_dataset)

test_in_normalizer = provide_normalizer(test_dataset, "x", 30)
# test_in_normalizer = UnitGaussianNormalizer.from_dataset(
#     dim=[0, 2, 3],
#     dataset = test_dataset,
#     mask = validity_preserving_input_normalize_mask,
#     keys=["x"]
# )

test_in_normalizer = provide_normalizer(test_dataset, "y", 4)
# test_out_normalizer = UnitGaussianNormalizer.from_dataset(
#     dim=[0, 2, 3],
#     dataset = test_dataset,
#     mask = validity_preserving_output_normalize_mask,
#     keys=["y"]
# )

# Create the FNO model
# ----------------------

model = FNO(
    n_modes=(12, 12),   #this tuple has to be as long as the temporal axes
    in_channels=3*5*2,     # = length of input channel - 3 intervals * 5 attributes * infolded value/validity dimension
    out_channels=2*2,     # = number of output variables - 2 intervals * infolded value/validity dimension
    hidden_channels=32,     # hyperparameter
    positional_embedding="grid"  #ABSOLUTELY FIX THIS LATER?
)
model = model.to(device)

# Count and display the number of parameters
n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()


# # Creating the optimizer and scheduler
# # ------------------------------------
# # We use AdamW optimizer with weight decay for regularization
optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

train_loss = MSELoss()
eval_losses = {"MSE": MSELoss() }

# # Training the model
# # ---------------------
# # We display the training configuration and then train the model

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

# # Creating the trainer
# # --------------------
# # We create a Trainer object that handles the training loop, evaluation, and logging
trainer = Trainer(
    model=model,
    n_epochs=epochs,
    device=device,
    data_processor= train_processor,
    wandb_log=False,  # Disable Weights & Biases logging for this tutorial
    eval_interval=5,  # Evaluate every 5 epochs
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,  # Print training progress
)

# # Training the model
# # ------------------
# # We train the model on our Darcy-Flow dataset. The trainer will:
# # 1. Run the forward pass through the FNO
# # 2. Compute the H1 loss
# # 3. Backpropagate and update weights
# # 4. Evaluate on test data every 3 epochs

test_loaders = {"one": test_loader}
print("Training")

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

model.save_checkpoint(os.path.dirname(ckpt_path), os.path.basename(ckpt_path))


# Visualizing predictions
# ------------------------
# Let's take a look at what our model's predicted outputs look like.
# We wll compare the inputs, ground-truth outputs, and model predictions side by side.
#
# Note that in this example, we train on a very small resolution for
# a very small number of epochs. In practice, we would train at a larger
# resolution on many more samples.

test_samples = test_loaders["one"].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input
    x = data["x"]
    # Ground-truth output
    y = data["y"]
    # Model prediction
    out = model(x.unsqueeze(0))

    # # Plot input
    # ax = fig.add_subplot(3, 3, index * 3 + 1)
    # ax.imshow(x[0], cmap="gray")
    # if index == 0:
    #     ax.set_title("Input x")
    # plt.xticks([], [])
    # plt.yticks([], [])

    # Plot ground-truth output
    # ax = fig.add_subplot(3, 3, index * 3 + 2)
    # ax.imshow(y.squeeze())
    # if index == 0:
    #     ax.set_title("Ground-truth output")
    # plt.xticks([], [])
    # plt.yticks([], [])

    # Plot model prediction
    # ax = fig.add_subplot(3, 3, index * 3 + 3)
    # ax.imshow(out.squeeze().detach().numpy())
    # if index == 0:
    #     ax.set_title("Model prediction")
    # plt.xticks([], [])
    # plt.yticks([], [])

# fig.suptitle("FNO predictions on 16x16 Darcy-Flow data", y=0.98)
# plt.tight_layout()
# fig.show()