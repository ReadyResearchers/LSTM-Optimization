"""
Optuna example that optimizes LSTM hyperparameters using PyTorch.

Source Code Credit: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import sys

# For performance reasons, if using the cpu, lower the number of epochs
# And if using the gpu, raise the number of epochs
DEVICE = torch.device("cpu")


class Dataset(Dataset):
    """Build the dataset based on a csv file of data."""

    # Dataframes are fields shared amongst functions
    input_df = None
    output_df = None

    def __init__(self):
        """Create the instance of the Dataset, initializing the input and the target output"""

        # Create pandas dataframes from a csv file and set up the input and output dataframes based on CLI arguments
        orig_df = pd.read_csv(sys.argv[1])
        input_keys = map(str, sys.argv[2].strip("[]").split(","))
        output_key = map(str, sys.argv[3].strip("[]").split(","))
        self.input_df = orig_df[input_keys]
        self.output_df = orig_df[output_key]

        # Convert the dataframes to numpy arrays
        input_np = self.input_df.to_numpy()
        output_np = self.output_df.to_numpy().reshape(-1, 1)

        # Construct the input and output arrays
        input_data = []
        output_data = []
        projection = []
        input_seq_length = int(sys.argv[4])
        output_seq_length = int(sys.argv[5])

        for i in range(input_seq_length, input_np.shape[0] - output_seq_length):
            input_data.append(input_np[i - input_seq_length : i, :])
            for j in range(output_seq_length):
                if i + j < output_np.shape[0]:
                    projection.append(output_np[i + j, 0])
            output_data.append(projection)
            projection = []

        # Convert arrays into numpy arrays
        final_input = np.array(input_data)
        final_output = np.array(output_data)

        # Initialize the fields in the Dataset object
        self.x = final_input
        self.y = final_output
        self.n_samples = len(final_input)

    def __getitem__(self, index):
        """Return the item based on the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the quantity of data points."""
        return self.n_samples


def extract_data(trial):
    """Load the dataset to a DataLoader and visualize the data being collected."""
    # Suggest a batch size
    batch_size = trial.suggest_int("batch_size", 10, 100)

    # Two data loaders are created; one for training and the other for evaluation
    train_loader = DataLoader(data, batch_size=batch_size)
    valid_loader = DataLoader(data, batch_size=batch_size)

    return train_loader, valid_loader, batch_size


def define_model(trial):
    """Construct the model while optimizing the number of layers, hidden units and dropout rate."""
    # Configurations for the model, some of them suggested and others from CLI arguments
    n_layers = trial.suggest_int("n_layers", 2, 5)
    input_size = len(sys.argv[2].split(","))
    proj_size = int(sys.argv[5])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size = trial.suggest_int("hidden_size", proj_size + 1, 2048)

    # Return the entire model
    return nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=n_layers,
        batch_first=True,
        dropout=dropout,
        proj_size=proj_size,
    )


def objective(trial):
    """Simulate the neural networks and evaluate them."""

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical(
        "optimizer",
        [
            "Adam",
            "RMSprop",
            "SGD",
            "Adadelta",
            "NAdam",
            "ASGD",
            "Adagrad",
            "AdamW",
            "Adamax",
            "RAdam",
            "Rprop",
        ],
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset and batch size. 80-20 split for training and testing
    train_loader, valid_loader, batch_size = extract_data(trial)
    batches = len(train_loader.dataset) // batch_size
    test_batches = batches // 5

    # Training of the model.
    for _ in range(int(sys.argv[7])):

        # Put the model in training mode
        model.train()
        num_train_batches = batches - test_batches

        # This is in case the batch size is large relative to the dataset, leaving no room for the test batches.
        if test_batches == 0:
            num_train_batches = batches - 1
            test_batches = 1

        # Iteratively take batches of data and put them into the model
        for batch_idx, (x, y) in enumerate(train_loader):
            # Leave data for the validation process
            if batch_idx == num_train_batches:
                break

            # Convert the input and output tensors to have float values
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()

            # Min-max scaling
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())

            # Output from LSTM is the format of Tuple[Tensor, Tuple[Tensor, Tensor]]
            # The Tensor inside the inner Tuple is chosen because it just has the hidden values from the last time step
            output = model(x)[1][0]

            # Match the shape of the output from the model to the shape of the targets
            y = (
                y.unsqueeze(0)
                .expand(output.shape[0], y.shape[0], y.shape[1])
                .to(DEVICE)
            )

            # Calculate the loss
            optimizer.zero_grad()
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        # Put the model in validation mode
        model.eval()
        eval_entries = 0
        sqerror = 0
        with torch.no_grad():

            # Iteratively take batches of data and put them into the model
            for batch_idx, (x, y) in enumerate(valid_loader):
                # This condition is made so the evaluation is only made on untrained data.
                if batch_idx >= num_train_batches:
                    eval_entries += batch_size

                    # Convert the input tensor to a float
                    x = x.to(DEVICE).float()
                    y = y.to(DEVICE).float()

                    # Output from LSTM is the format of Tuple[Tensor, Tuple[Tensor, Tensor]]
                    # The Tensor inside the inner Tuple is chosen because it only has the hidden values from the last
                    # time step, which holds the projections
                    output = model(x)[1][0]

                    # Match the shape of the output from the model to the shape of the targets
                    y = (
                        y.unsqueeze(0)
                        .expand(output.shape[0], y.shape[0], y.shape[1])
                        .to(DEVICE)
                    )

                    # Add the error
                    loss_fn = nn.MSELoss()
                    sqerror += loss_fn(output, y)

                    # If the index in the next batch after this iteration goes out of bounds, exit the loop
                    if (batch_idx + 2) * batch_size >= len(valid_loader.dataset):
                        break

        # Compute the MSE which will be the value the study minimizes through different suggestions of hyperparameters
        mse = sqerror / eval_entries

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mse


if __name__ == "__main__":
    # Create a Dataset to collect the data, visualize the data, and set up the inputs and outputs to train the LSTM
    data = Dataset()

    # Activation of the study, aiming to minimize MSE
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(sys.argv[6]), timeout=10)

    with open("experiment/best_values.pickle", "wb") as f:
        # Use pickle to dump the dictionary of the trial into the file
        pickle.dump(study.best_trial, f)

    # Display information about the best trial
    print("Best trial:")
    trial = study.best_trial

    print("  MSE: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
