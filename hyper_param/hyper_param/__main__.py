"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of a stock called Britannica using
PyTorch and a csv file containing the stock data. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time-consuming to use the whole file,
we here use a small subset of it.

Source Code Credit: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
Stock Market Data Credit: https://www.kaggle.com/code/jagannathrk/stock-market-time-series/data?select=BRITANNIA.csv
Temperature Data Credit: https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities?resource=download
"""

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import sys

DEVICE = torch.device("cpu")
BATCHSIZE = 200
CLASSES = 1
EPOCHS = 50
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


class Dataset(Dataset):
    """Build the dataset based on a csv file of data."""

    def __init__(self):
        orig_df = pd.read_csv(sys.argv[1])
        input_keys = map(str, sys.argv[2].strip('[]').split(','))
        output_key = map(str, sys.argv[3].strip('[]').split(','))
        input_df = orig_df[input_keys]
        output_df = orig_df[output_key]

        num_plots = input_df.shape[1] + output_df.shape[1]
        plot_rows = num_plots
        plot_cols = 1

        while plot_rows > 10:
            plot_cols += 1
            plot_rows = math.ceil(num_plots / plot_cols)

        fig = make_subplots(rows=plot_rows, cols=plot_cols)

        x = 0
        i = 1
        j = 1

        while x < input_df.shape[1]:
            fig.add_trace(go.Scatter(x=input_df.index, y=input_df[input_df.columns[x]].values,
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

        fig.add_trace(go.Scatter(x=output_df.index, y=output_df[output_df.columns[x]].values,
                                 name=output_df.columns[x],
                                 mode='lines'),
                      row=i,
                      col=j)

        fig.update_layout(height=1200, width=1200)
        fig.show()

        input_np_unscaled = input_df.to_numpy()
        output_np_unscaled = output_df.to_numpy().reshape(-1, 1)

        scaler_train = MinMaxScaler()
        input_np_scaled = scaler_train.fit_transform(input_np_unscaled)
        output_np_scaled = scaler_train.fit_transform(output_np_unscaled)

        self.x = torch.from_numpy(input_np_scaled)
        self.y = torch.from_numpy(output_np_scaled)
        self.n_samples = input_df.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def extract_data():
    """Load stock dataset in a DataLoader and visualize the data being collected."""
    train_loader = DataLoader(Dataset(), batch_size=BATCHSIZE)
    valid_loader = DataLoader(Dataset(), batch_size=BATCHSIZE)

    return train_loader, valid_loader


def define_model(trial):
    """Construct the model while optimizing the number of layers, hidden units and dropout ratio in each layer."""
    n_layers = trial.suggest_int("n_layers", 1, 10)
    layers = []

    in_features = 6

    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 2, 100)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, CLASSES))

    return nn.Sequential(*layers)


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

    # Get the dataset.
    train_loader, valid_loader = extract_data()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = torch.flatten(model(data))
            target = torch.flatten(target.type(torch.LongTensor))
            loss = F.mse_loss(
                torch.nn.functional.relu(output).float(), torch.nn.functional.relu(target).float()
            )
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        sqerror = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                sqerror += torch.sum(torch.square(torch.sub(output, target)))

        rmse = (sqerror / min(len(valid_loader.dataset), N_VALID_EXAMPLES)) ** (0.5)

        trial.report(rmse, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return rmse


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10000, timeout=10000)

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

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
