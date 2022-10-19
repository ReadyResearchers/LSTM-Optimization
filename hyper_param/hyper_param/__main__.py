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
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cpu")
BATCHSIZE = 200
CLASSES = 2
EPOCHS = 50
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


class StockDataset(Dataset):
    """Build the dataset based on a csv file of stock data."""

    def __init__(self):
        xy = np.loadtxt(
            "./data/BRITANNIA.csv", delimiter=",", dtype=np.float32, skiprows=1
        )
        closing_price = xy[:, [0, 5]]
        self.x = torch.from_numpy(xy[:, [0, 6, 7, 8, 9, 10]])
        self.y = torch.from_numpy(
            np.less(closing_price[:, 0], closing_price[:, 1])
        ).type(torch.LongTensor)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class TempDataset(Dataset):
    """Build the dataset based on a csv file of temperature data."""

    def __init__(self):
        xy = pd.read_csv("./data/city_temperature_compressed.csv", dtype=str)
        xy[["Region", "Country", "City"]] = xy[["Region", "Country", "City"]].apply(
            LabelEncoder().fit_transform
        )
        self.x = (
            xy.loc[:, ["Region", "Country", "City", "Month", "Day"]]
            .to_numpy(dtype="float32")
            .astype(np.float32)
        )
        self.y = (
            xy.loc[:, ["AvgTemperature"]].to_numpy(dtype="float32").astype(np.int32)
        )
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def get_stock_data():
    """Load stock dataset in a DataLoader so that the neural networks are trained based on batches of data."""
    train_loader = DataLoader(StockDataset(), batch_size=BATCHSIZE, shuffle=True)
    valid_loader = DataLoader(StockDataset(), batch_size=BATCHSIZE, shuffle=True)

    return train_loader, valid_loader


def get_temp_data():
    """Load temperature dataset in a DataLoader so that the neural networks are trained based on batches of data."""
    train_loader = DataLoader(TempDataset(), batch_size=BATCHSIZE, shuffle=True)
    valid_loader = DataLoader(TempDataset(), batch_size=BATCHSIZE, shuffle=True)

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

    # Get the stock dataset.
    train_loader, valid_loader = get_stock_data()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            target = torch.flatten(target.type(torch.LongTensor))
            loss = F.nll_loss(
                torch.nn.functional.relu(output), torch.nn.functional.relu(target)
            )
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

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
