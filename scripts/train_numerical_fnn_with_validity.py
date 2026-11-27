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
from inspect_processed_data import get_rain_data_and_validity_ndarray, get_numerical_data_and_validity_ndarray, get_numerical_data_ndarray, get_rain_data_ndarray
from weather_dataset import NumericalWeatherDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor


device = "cpu"

import numpy as np


# create a dataset for the training samples
train_numerical_sample_tensor = torch.from_numpy(get_numerical_data_and_validity_ndarray(0, 16)).type(torch.FloatTensor)
train_rain_sample_tensor = torch.from_numpy(get_rain_data_and_validity_ndarray(0, 16)).type(torch.FloatTensor)
print(type(train_numerical_sample_tensor), type(train_rain_sample_tensor))
print(train_numerical_sample_tensor.shape, train_rain_sample_tensor.shape)

train_dataset = NumericalWeatherDataset(train_numerical_sample_tensor, train_rain_sample_tensor)
train_loader = DataLoader(train_dataset)

#create a normalizer for the inputs
#prevent the validity mask from being normalized
validity_preserving_input_normalize_mask = np.array([np.ones((3 * 5, 32, 32)), np.zeros((3 * 5,32,32))])
validity_preserving_input_normalize_mask = np.transpose(validity_preserving_input_normalize_mask, (1, 0, 2, 3))
attr, v, x, y = np.shape(validity_preserving_input_normalize_mask)
validity_preserving_input_normalize_mask = np.reshape(validity_preserving_input_normalize_mask, (attr * v, x, y))

train_in_normalizer = UnitGaussianNormalizer.from_dataset(
    dataset = train_dataset,
    mask = validity_preserving_input_normalize_mask,
    keys=["x"]
)

#create a normalizer for the outputs
validity_preserving_output_normalize_mask = np.array([np.ones((2, 32, 32)), np.zeros((2,32,32))])
validity_preserving_output_normalize_mask = np.transpose(validity_preserving_output_normalize_mask, (1, 0, 2, 3))
attr, v, x, y = np.shape(validity_preserving_output_normalize_mask)
validity_preserving_output_normalize_mask = np.reshape(validity_preserving_output_normalize_mask, (attr * v, x, y))

train_out_normalizer = UnitGaussianNormalizer.from_dataset(
    dataset = train_dataset,
    mask = validity_preserving_output_normalize_mask,
    keys=["y"]
)

train_processor = DefaultDataProcessor(in_normalizer=train_in_normalizer, out_normalizer=train_out_normalizer)

#do the same but for the testing data

test_numerical_sample_tensor = torch.from_numpy(get_numerical_data_and_validity_ndarray(16, 20)).type(torch.FloatTensor)
test_rain_sample_tensor = torch.from_numpy(get_rain_data_and_validity_ndarray(16, 20)).type(torch.FloatTensor)
print(type(test_numerical_sample_tensor), type(test_rain_sample_tensor))
print(test_numerical_sample_tensor.shape, test_rain_sample_tensor.shape)

test_dataset = NumericalWeatherDataset(test_numerical_sample_tensor, test_rain_sample_tensor)
test_loader = DataLoader(test_dataset)

test_in_normalizer = UnitGaussianNormalizer.from_dataset(
    dataset = test_dataset,
    mask = validity_preserving_input_normalize_mask,
    keys=["x"]
)

test_out_normalizer = UnitGaussianNormalizer.from_dataset(
    dataset = test_dataset,
    mask = validity_preserving_output_normalize_mask,
    keys=["y"]
)

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

# # Setting up loss functions
# # -------------------------
# # We use H1 loss for training and L2 loss for evaluation
# # H1 loss is particularly good for PDE problems as it penalizes both function values and gradients
l2loss = LpLoss(d=2, p=2)  # L2 loss for function values
h1loss = H1Loss(d=2)  # H1 loss includes gradient information

# mean_squared_error_loss = MSELoss()

train_loss = MSELoss()
eval_losses = {"h1": h1loss }

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
    n_epochs=15,
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

#print(train_loader.dataset.__getitem__(0)["y"])

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)


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