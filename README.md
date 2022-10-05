# MLP-Optimization

![build](https://github.com/ReadyResearchers/MLP-Optimization/actions/workflows/build.yml/badge.svg)

## Introduction

For my research project, I'll intend to use neural networks to perform predictive
analytics on time series datasets, but mainly focusing on performing hyperparameter
optimization on neural networks that are made based on the dataset being analyzed
and the interpretation objective. The goal is to find trends of the most optimal
hyperparameters based on variations in the properties of data and the
interpretation objective.

## Technical Details

The packages being used for this artifact are called `optuna` and `PyTorch`.
`optuna` is a hyperparameter optimization framework for the Python language.
`Pytorch` is a machine learning framework for the Python language that
specializes in the creation of neural networks for computer vision and
natural language processing.

In this application of hyperparameter optimization, I try to analyze a particular
stock in a csv file and then run through trials to come up with the some of the best
hyperparameters within a range of values. This neural network would try to
predict one of two outputs, which are whether the closing price goes up or down.

First, `optuna` creates a study that stores all of the trials and their results.

At the beginning of a trial run, a model for the neural network is created.

The number of layers and the number of neurons in each layer are decided by the
`suggest_int` function, which picks a hyperparameter and then chooses the value of
that hyperparameter based on potential.

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

Once all of the trials are ran, the most optimal trial is displayed, with the
accuracy and the hyperparameters that are chosen for that trial.

## Installation and Setup

### Python

You'll need to have Python 3.10 or later for this program to work on your machine.
If you haven't installed it yet, visit the [official website](https://www.python.org)
and make sure to add the path of the Python binary to the environment so that
it's recognized.

### Poetry

Once Python is installed, you'll need to also install Poetry, which is a
Python tool that will hold and manage all of the packages that are needed
for this artifact. All of the information needed to install Poetry in
your system is in [this link](https://python-poetry.org/docs/).

### Setup

Once Poetry is installed, you're now able to run the program by first entering
`poetry install` in the `hyper_param` directory to install the packages
and then entering `poetry run python hyper_param` to run the program itself.

## Related Works

[Neural Networks for COVID Projections](https://www.sciencedirect.com/science/article/pii/S2772662221000060)

[Neural Network Parameter Optimization](https://www.sciencedirect.com/science/article/abs/pii/S0925231215020184?casa_token=RXOg711Fbs0AAAAA:KJsnEcjVitIX6KTRR0W88cmcuomo1-oGHGbZpk4jlphHwuk7SNpg48bX0zwLw9THn9Ibv0R9UQ)

[Prediction of RUL of Equipment in Production Lines using ANN](https://www.mdpi.com/1424-8220/21/3/932)

## Future Endeavors

Try out new datasets.

Try to come up with appropriate ranges of hyperparameters that are suitable for
projections.

Decide whether to completely change course on the packages used for constructing
a neural network due to the output following categorical tendencies.

Make visualizations that aim at contributing to the findings of the tendencies
of the most optimal hyperparameters.