"""
Optuna example that optimizes LSTM using PyTorch.
In this example, we optimize the projection accuracy of a stock called Britannica using
PyTorch and a csv file containing the stock data. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time-consuming to use the whole file,
we here use one small batch of it at a time.

Source Code Credit: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
Stock Market Data Credit: https://www.kaggle.com/code/jagannathrk/stock-market-time-series/data?select=BRITANNIA.csv
Temperature Data Credit: https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities?resource=download
"""

import optuna
import optuna.visualization as vis
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import sys

# For performance reasons, if using the cpu, lower the number of epochs
# And if using the gpu, raise the number of epochs
DEVICE = torch.device("cuda")
EPOCHS = 10


class Dataset(Dataset):
    """Build the dataset based on a csv file of data."""
    # Dataframes are fields shared amongst functions
    input_df = None
    output_df = None
    def __init__(self):
        """Create the instance of the Dataset, initializing the input and the target output"""

        # Create pandas dataframes from a csv file and set up the input and output dataframes based on CLI arguments
        orig_df = pd.read_csv(sys.argv[1])
        input_keys = map(str, sys.argv[2].strip('[]').split(','))
        output_key = map(str, sys.argv[3].strip('[]').split(','))
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
            input_data.append(input_np[i - input_seq_length:i, :])
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

    def visualize(self):
        """Display the collected data."""
        input_df = self.input_df
        output_df = self.output_df

        # Determine how many plots to make
        num_plots = input_df.shape[1] + output_df.shape[1]

        # Determine how many rows and columns are needed for the amount of plots
        plot_rows = num_plots
        plot_cols = 1

        # If there are many rows, have multiple columns so all the plots fit
        while plot_rows > 10:
            plot_cols += 1
            plot_rows = math.ceil(num_plots / plot_cols)

        # Set up the container of the plots with the number of rows and columns
        data_fig = make_subplots(rows=plot_rows, cols=plot_cols)

        # Add the plots of each individual input and output column to the container.
        # Locations of these plots have to be manually specified
        x = 0
        i = 1
        j = 1

        while x < input_df.shape[1]:
            data_fig.add_trace(go.Scatter(x=input_df.index, y=input_df[input_df.columns[x]].values,
                                          name=input_df.columns[x],
                                          mode='lines'),
                               row=i,
                               col=j)
            x += 1
            j += 1
            if j > plot_cols:
                j = 1
                i += 1

        x = 0

        data_fig.add_trace(go.Scatter(x=output_df.index, y=output_df[output_df.columns[x]].values,
                                      name=output_df.columns[x],
                                      mode='lines'),
                           row=i,
                           col=j)

        # Display the container of plots
        data_fig.update_layout(height=1200, width=1200)
        data_fig.show()



def extract_data(trial):
    """Load the dataset to a DataLoader and visualize the data being collected."""
    # Suggest a batch size
    batch_size = trial.suggest_int('batch_size', 20, 400)

    # Two data loaders are created; one for training and the other for evaluation
    train_loader = DataLoader(data, batch_size=batch_size)
    valid_loader = DataLoader(data, batch_size=batch_size)

    return train_loader, valid_loader, batch_size


def define_model(trial):
    """Construct the model while optimizing the number of layers, hidden units and dropout ratio."""
    # Configurations for the model, some of them suggested and others from CLI arguments
    n_layers = trial.suggest_int('n_layers', 2, 10)
    input_size = len(sys.argv[2].split(','))
    proj_size = int(sys.argv[5])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    hidden_size = trial.suggest_int('hidden_size', proj_size + 1, 512)

    # Return the entire model
    return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                   dropout=dropout, proj_size=proj_size)


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
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset and batch size.
    train_loader, valid_loader, batch_size = extract_data(trial)
    test_batches = trial.suggest_int('test_batches', 1, 10)

    # Training of the model.
    for epoch in range(EPOCHS):

        # Put the model in training mode
        model.train()
        num_train_batches = len(train_loader.dataset) // batch_size - test_batches

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
            y = y.unsqueeze(0).expand(output.shape[0], y.shape[0], y.shape[1]).to(DEVICE)

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
                    y = y.unsqueeze(0).expand(output.shape[0], y.shape[0], y.shape[1]).to(DEVICE)

                    # Add the error
                    loss_fn = nn.MSELoss()
                    sqerror += loss_fn(output, y)

                    # If the index in the next batch after this iteration goes out of bounds, exit the loop
                    if (batch_idx + 2) * batch_size >= len(valid_loader.dataset):
                        break

        # Compute the MSE which will be the value the study minimizes through different suggestions of hyperparameters
        mse = sqerror / eval_entries

        trial.report(mse, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mse


if __name__ == "__main__":
    # Create a Dataset to collect the data, visualize the data, and set up the inputs and outputs to train the LSTM
    data = Dataset()
    data.visualize()

    # Activation of the study, aiming to minimize AMSE
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10000, timeout=120)

    # Evaluation of the study
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Visualization
    vis.plot_param_importances(study).show()
    vis.plot_contour(study).show()
