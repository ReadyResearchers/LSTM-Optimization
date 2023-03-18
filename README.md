# LSTM-Optimization

![logo](images/logo.png)
![build](https://github.com/ReadyResearchers/MLP-Optimization/actions/workflows/build.yml/badge.svg)
[![codecov](https://codecov.io/gh/ReadyResearchers/LSTM-Optimization/branch/main/graph/badge.svg?token=KI26GOFV8B)](https://codecov.io/gh/ReadyResearchers/LSTM-Optimization)

## Table of Contents

* [Title of Senior Thesis](#title-of-senior-thesis)
* [Introduction](#introduction)
* [Abstract](#abstract)
* [Technical Details](#technical-details)
* [Data Analysis](#data-analysis)
* [Reproducibility Details](#reproducibility-details)
  + [Python](#python)
  + [CPU](#cpu)
  + [GPU](#gpu)
* [Related Works](#related-works)
* [Future Endeavors](#future-endeavors)

## Title of Senior Thesis

Digital Prophet: Predictive Analytics of Time Series Datasets Using
Hyperparameter Optimization

## Introduction

For my research project, I'll intend to use neural networks to perform predictive
analytics on time series datasets, but mainly focusing on performing hyperparameter
optimization on neural networks that are made based on the dataset being analyzed
and the interpretation objective. The goal is to find trends of the most optimal
hyperparameters based on variations in the properties of data and the
interpretation objective.

## Abstract

LSTM-OP is designed to optimize the configuration of neural networks through
the use of hyperparameter optimization. Neural networks are widely used in
various industries due to their ability to identify complex patterns in data and
make accurate predictions. They are particularly useful in scenarios where it is
difficult for humans to accurately analyze and predict outcomes based on the
available data. However, finding the optimal configuration of a neural network
can be a challenging process, as it often involves trade-offs between performance
and optimality. This is where a series of findings generated by LSTM-OP comes in.
By using hyperparameter optimization, it is possible to mitigate these trade-offs
and efficiently optimize the configuration of the neural network. This can be
especially useful in scenarios where the objective function is difficult to
differentiate. This series of findings is meant to achieve this goal,
significantly reducing the number of trial and error iterations typically required
to find the optimal configuration while still maximizing performance.

## Technical Details

The packages being used for this artifact are called `optuna` and `PyTorch`.
`optuna` is a hyperparameter optimization framework for the Python language.
`Pytorch` is a machine learning framework for the Python language that
specializes in the creation of neural networks for computer vision and
natural language processing.

In this application of hyperparameter optimization, I analyze multiple time series
datasets in a csv format and then for each dataset run through trials to come up
with the most optimal hyperparameter values within a certain range.

First, `optuna` creates a study that stores all the trials and their results.

At the beginning of a trial run, a model for the neural network is created.

The number of layers and the number of neurons in each layer are decided by the
`suggest_<data_type>` function, which creates a hyperparameter and chooses
the value of that hyperparameter based on potential.

The optimizer is another hyperparameter which value is chosen through the
`suggest_categorical` function.

Then the actual data comes in. This is done through a `DataLoader`, which divides
the dataset into batches for the neural network to be trained from.

Then the simulation of the neural network starts. Through each epoch, the loss is
calculated, and then once the epochs are run, the accuracy is established and
displayed, which completes the trial.

There is the most optimal trial that's established, and as more trials are being
ran, the hyperparameters are chosen to try to get a more accurate model,
and if the loss function doesn't go down fast enough to catch up with the
mostoptimal trial, then it gets pruned, which is the process of just ending the
trial early and going to the next one.

Once all the trials are ran, the most optimal trial is displayed, with the
mean square error and the hyperparameter values chosen for that trial.

## Data Analysis

In the data directory, there are quite a few files. For each dataset, the
seasonality strength had to be verified, so I created a series of
autocorrelation functions through the use of R scripts. The datasets,
plots, and the experiment results are all in the data directory. The
experiment directory has an analysis directory that holds all the R scripts.

## Reproducibility Details

There are two primary ways to run the program. One way is to directly run
LSTM-OP, and another way is to run the experiment which run LSTM-OP many times.
There are also different mechanisms of execution depending on whether the
CPU or GPU is used.

### Python

You'll need to have Python 3.10 or later for this program to work on your machine.
If you haven't installed it yet, visit the [official website](https://www.python.org)
and make sure to add the path of the Python binary to the environment so that
it's recognized.

### CPU

#### Experiment for CPU

In the "lstm.py" file, there is a final variable called `DEVICE`. Make sure
that the string inside the device says `cpu`, not `cuda`. Once the latest
version of Poetry is installed, navigate to the root of the repository and then
to a directory called "hyper_param". When inside the directory, enter the
following command: `poetry run python experiment`.

#### Direct for CPU

Running LSTM-OP without using the script is slightly different. It's similar
to the previous command, but there needs to be additional arguments. An
example of a command you would run is the following: `poetry run python
hyper_param/lstm.py data/city_temperature_compressed.csv [Year,Month,Day]
[AvgTemperature] 10 10 50 10`. Command line arguments are as follows: file
path of the dataset, list of input features, output feature, amount of input
time steps, amount of output time steps, number of trials, and number of epochs.

### GPU

#### Experiment for GPU

In the `lstm.py` file, there is a final variable called `DEVICE`. Make
sure that the string inside the device says `cuda`, not `cpu`. Once a CUDA
version of 11.7 or later is successfully installed, Conda needs to also be
installed. The simplest way to do this is to install Anaconda, which is a
distribution that already has the Conda package manager. Once Anaconda is
installed, open the Anaconda terminal and make sure all the packages in
the manager are updated by entering the following command: `conda update --all`.
After that, refer to the `pyproject.toml` file that lists the packages Conda
needs to install. Then for each package under the `tool.poetry.dependencies`
block, enter the command: `conda install <package-name>`. After that's done,
navigate to the root of the repository and then to a directory called
`hyper_param`. When inside the directory, enter the following command:
`python experiment/__main__.py`.

#### Direct for GPU

Running LSTM-OP without using the script is slightly different. It's
similar to the previous command, but there needs to be additional
arguments. An example of a command you would run is the following:
`python hyper_param/lstm.py data/city_temperature_compressed.csv
[Year,Month,Day] [AvgTemperature] 10 10 50 10`. Command line arguments
are as follows: file path of the dataset, list of input features,
output feature, amount of input time steps, amount of output time
steps, number of trials, and number of epochs.

## Related Works

[Neural Networks for COVID Projections](https://www.sciencedirect.com/science/article/pii/S2772662221000060)

[Neural Network Parameter Optimization](https://www.sciencedirect.com/science/article/abs/pii/S0925231215020184?casa_token=RXOg711Fbs0AAAAA:KJsnEcjVitIX6KTRR0W88cmcuomo1-oGHGbZpk4jlphHwuk7SNpg48bX0zwLw9THn9Ibv0R9UQ)

[Prediction of RUL of Equipment in Production Lines using ANN](https://www.mdpi.com/1424-8220/21/3/932)

[Time Series Forecasting Models](https://otexts.com/fpp2/data-methods.html)

## Future Endeavors

Find a way to fake the datasets so that they fit a certain value,
such as the slope of the overall trend that results from decomposition.

Do the experiment in a more exhaustive manner.
