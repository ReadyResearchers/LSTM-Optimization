# MLP-Optimization

## Introduction

For my research project, I'll intend to use neural networks to perform predictive analytics on time series datasets. I'll 
also perform hyperparameter optimization on neural networks that are made based on the dataset being analyzed and the 
interpretation objective. The goal is to find trends of the most optimal hyperparameters based on variations in the 
properties of data and the interpretation objective.

## Technical Details

The packages being used for this artifact are called `optuna` and `PyTorch`. `optuna` is a hyperparameter optimization framework for the Python language. `Pytorch` is a machine learning framework for the Python language that specializes in the creation of neural networks for computer vision and natural language processing.

In this application of hyperparameter optimization, I try to analyze a particular stock in a csv file and then run through trials to come up with the some of the best hyperparameters within a range of values. This neural network would try to predict one of two outputs, which are whether the closing price goes up or down.

- First, `optuna` creates a study that stores all of the trials and their results.
- At the beginning of a trial run, a model for the neural network is created. The number of layers and the number of neurons in each layer are decided by the `suggest_int` function, which picks a hyperparameter and then chooses the value of that hyperparameter based on potential.
- The optimizer is another hyperparameter which value is chosen through the `suggest_categorical` function.
- Then the actual data comes in. This is done through a `DataLoader`, which divides the dataset into batches for the neural network to be trained from.
- Then the simulation of the neural network starts. Through each epoch, the loss is calculated, and then once the epochs are run through, the accuracy is established and displayed, and that completes the trial.
- There is the most optimal trial that's established, and as more trials are being ran, the hyperparameters are chosen to try to get more accurate, and if the loss function doesn't go down fast enough to catch up with the most optimal trial, then it gets pruned, which the process of just ending the trial early and going to the next one.
- Once all of the trials are ran, the most optimal trial is displayed, with the accuracy and the hyperparameters that are chosen for that trial.

## Related Works

## Future Endeavors