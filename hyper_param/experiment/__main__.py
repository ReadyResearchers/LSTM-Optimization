import pickle
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objs as go
import os
import sys

if __name__ == "__main__":

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Controls: 10 epochs, 10 input and output time steps, 50 trials

    # List of possible values for each command line argument
    epochs = [5, 8, 10, 20, 30]
    time_steps = [3, 8, 10, 15, 20]
    projection_size = [5, 8, 10, 15, 20]
    trials = [10, 25, 50, 75, 100]

    # Initialize the list of mean square errors and optimization algorithms for each permutation
    mse_list = []
    optimizers = []

    # Construction of the collection of subplots for the entire experiment
    data_fig = make_subplots(rows=4, cols=1)

    # Always make the lists hyperparameter values empty when starting a new subplot
    y_batch_size = []
    y_lr = []
    y_layers = []
    y_dropout = []
    y_hidden_size = []

    # Customization of the lines
    trace_batch_size = go.Scatter(x=epochs, y=y_batch_size, name="Batch Size", mode="lines", line=dict(color="blue"))
    trace_lr = go.Scatter(x=epochs, y=y_lr, name="Learning Rate", mode="lines", line=dict(color="red"))
    trace_layers = go.Scatter(x=epochs, y=y_layers, name="Number of Layers", mode="lines", line=dict(color="green"))
    trace_dropout = go.Scatter(x=epochs, y=y_dropout, name="Dropout Rate", mode="lines", line=dict(color="orange"))
    trace_hidden_size = go.Scatter(x=epochs, y=y_hidden_size, name="Hidden Size", mode="lines", line=dict(color="black"))

    for e in epochs:
        # Initialize each command line argument being passed into LSTM-OP
        program = "hyper_param/lstm.py"
        csv = "data/jena_climate_data.csv"
        input = "[Year,Month,Day,Hour,p..mbar.,rh....,VPmax..mbar.,VPact..mbar.,VPdef..mbar.,sh..g.kg.," \
                "H2OC..mmol.mol.,rho..g.m..3.,wv..m.s.,max..wv..m.s.,wd..deg.]"
        output = "[T..degC.]"
        control_time_steps = 10
        control_projection_size = 10
        control_trials = 50
        test_epochs = e
        command = f"python {program} {csv} {input} {output} {control_time_steps} {control_projection_size}" \
                  f" {control_trials} {test_epochs}"
        print(command)

        # Pass the command into LSTM-OP
        os.system(command)

        # Load the data of the best trial from the pickle file
        with open("experiment/best_values.pickle", "rb") as f:
            best_trial = pickle.load(f)

        # Add to the lists of mean square errors and optimization algorithms
        mse_list.append(best_trial.value)
        optimizers.append(best_trial.params["optimizer"])

        # Add the normalized value of the hyperparameters
        y_batch_size.append((best_trial.params["batch_size"] - 10) / (100 - 10))
        y_lr.append((best_trial.params["lr"] - 0.00001) / (0.01 - 0.00001))
        y_layers.append((best_trial.params["n_layers"] - 2) / (5 - 2))
        y_dropout.append((best_trial.params["dropout"] - 0.1) / (0.5 - 0.1))
        y_hidden_size.append((best_trial.params["hidden_size"] - control_projection_size - 1) /
                             (2048 - control_projection_size - 1))

    # Update the value of the traces
    trace_batch_size.update(y=y_batch_size)
    trace_lr.update(y=y_lr)
    trace_layers.update(y=y_layers)
    trace_dropout.update(y=y_dropout)
    trace_hidden_size.update(y=y_hidden_size)

    # Put those updated values in the subplot
    data_fig.add_trace(trace_batch_size, row=1, col=1)
    data_fig.add_trace(trace_lr, row=1, col=1)
    data_fig.add_trace(trace_layers, row=1, col=1)
    data_fig.add_trace(trace_dropout, row=1, col=1)
    data_fig.add_trace(trace_hidden_size, row=1, col=1)

    # Reset the hyperparameter lists
    y_batch_size = []
    y_lr = []
    y_layers = []
    y_dropout = []
    y_hidden_size = []

    # Same process with the time steps
    for t in time_steps:
        program = "hyper_param/lstm.py"
        csv = "data/jena_climate_data.csv"
        input = "[Year,Month,Day,Hour,p..mbar.,rh....,VPmax..mbar.,VPact..mbar.,VPdef..mbar.,sh..g.kg.," \
                "H2OC..mmol.mol.,rho..g.m..3.,wv..m.s.,max..wv..m.s.,wd..deg.]"
        output = "[T..degC.]"
        test_time_steps = t
        control_projection_size = 10
        control_trials = 50
        control_epochs = 10
        command = f"python {program} {csv} {input} {output} {test_time_steps} {control_projection_size} " \
                  f"{control_trials} {control_epochs}"
        print(command)
        os.system(command)

        with open('experiment/best_values.pickle', 'rb') as f:
            best_trial = pickle.load(f)

        mse_list.append(best_trial.value)
        optimizers.append(best_trial.params["optimizer"])

        y_batch_size.append((best_trial.params["batch_size"] - 10) / (100 - 10))
        y_lr.append((best_trial.params["lr"] - 0.00001) / (0.01 - 0.00001))
        y_layers.append((best_trial.params["n_layers"] - 2) / (5 - 2))
        y_dropout.append((best_trial.params["dropout"] - 0.1) / (0.5 - 0.1))
        y_hidden_size.append((best_trial.params["hidden_size"] - control_projection_size - 1) /
                             (2048 - control_projection_size - 1))

    trace_batch_size.update(x=time_steps, y=y_batch_size, showlegend=False)
    trace_lr.update(x=time_steps, y=y_lr, showlegend=False)
    trace_layers.update(x=time_steps, y=y_layers, showlegend=False)
    trace_dropout.update(x=time_steps, y=y_dropout, showlegend=False)
    trace_hidden_size.update(x=time_steps, y=y_hidden_size, showlegend=False)

    data_fig.add_trace(trace_batch_size, row=2, col=1)
    data_fig.add_trace(trace_lr, row=2, col=1)
    data_fig.add_trace(trace_layers, row=2, col=1)
    data_fig.add_trace(trace_dropout, row=2, col=1)
    data_fig.add_trace(trace_hidden_size, row=2, col=1)

    y_batch_size = []
    y_lr = []
    y_layers = []
    y_dropout = []
    y_hidden_size = []

    # Same process with the projection size
    for p in projection_size:
        program = "hyper_param/lstm.py"
        csv = "data/jena_climate_data.csv"
        input = "[Year,Month,Day,Hour,p..mbar.,rh....,VPmax..mbar.,VPact..mbar.,VPdef..mbar.,sh..g.kg.," \
                "H2OC..mmol.mol.,rho..g.m..3.,wv..m.s.,max..wv..m.s.,wd..deg.]"
        output = "[T..degC.]"
        control_time_steps = 10
        test_projection_size = p
        control_trials = 50
        control_epochs = 10
        command = f"python {program} {csv} {input} {output} {control_time_steps} {test_projection_size} " \
                  f"{control_trials} {control_epochs}"
        print(command)
        os.system(command)

        with open('experiment/best_values.pickle', 'rb') as f:
            best_trial = pickle.load(f)

        mse_list.append(best_trial.value)
        optimizers.append(best_trial.params["optimizer"])

        y_batch_size.append((best_trial.params["batch_size"] - 10) / (100 - 10))
        y_lr.append((best_trial.params["lr"] - 0.00001) / (0.01 - 0.00001))
        y_layers.append((best_trial.params["n_layers"] - 2) / (5 - 2))
        y_dropout.append((best_trial.params["dropout"] - 0.1) / (0.5 - 0.1))
        y_hidden_size.append((best_trial.params["hidden_size"] - p - 1) / (2048 - p - 1))

    trace_batch_size.update(x=projection_size, y=y_batch_size)
    trace_lr.update(x=projection_size, y=y_lr)
    trace_layers.update(x=projection_size, y=y_layers)
    trace_dropout.update(x=projection_size, y=y_dropout)
    trace_hidden_size.update(x=projection_size, y=y_hidden_size)

    data_fig.add_trace(trace_batch_size, row=3, col=1)
    data_fig.add_trace(trace_lr, row=3, col=1)
    data_fig.add_trace(trace_layers, row=3, col=1)
    data_fig.add_trace(trace_dropout, row=3, col=1)
    data_fig.add_trace(trace_hidden_size, row=3, col=1)

    y_batch_size = []
    y_lr = []
    y_layers = []
    y_dropout = []
    y_hidden_size = []

    # Same process with the trials
    for t in trials:
        program = "hyper_param/lstm.py"
        csv = "data/jena_climate_data.csv"
        input = "[Year,Month,Day,Hour,p..mbar.,rh....,VPmax..mbar.,VPact..mbar.,VPdef..mbar.,sh..g.kg.," \
                "H2OC..mmol.mol.,rho..g.m..3.,wv..m.s.,max..wv..m.s.,wd..deg.]"
        output = "[T..degC.]"
        control_time_steps = 10
        control_projection_size = 10
        test_trials = t
        control_epochs = 10
        command = f"python {program} {csv} {input} {output} {control_time_steps} {control_projection_size} " \
                  f"{test_trials} {control_epochs}"
        print(command)
        os.system(command)

        with open('experiment/best_values.pickle', 'rb') as f:
            best_trial = pickle.load(f)

        mse_list.append(best_trial.value)
        optimizers.append(best_trial.params["optimizer"])

        y_batch_size.append((best_trial.params["batch_size"] - 10) / (100 - 10))
        y_lr.append((best_trial.params["lr"] - 0.00001) / (0.01 - 0.00001))
        y_layers.append((best_trial.params["n_layers"] - 2) / (5 - 2))
        y_dropout.append((best_trial.params["dropout"] - 0.1) / (0.5 - 0.1))
        y_hidden_size.append((best_trial.params["hidden_size"] - control_projection_size - 1) /
                             (2048 - control_projection_size - 1))

    trace_batch_size.update(x=trials, y=y_batch_size)
    trace_lr.update(x=trials, y=y_lr)
    trace_layers.update(x=trials, y=y_layers)
    trace_dropout.update(x=trials, y=y_dropout)
    trace_hidden_size.update(x=trials, y=y_hidden_size)

    data_fig.add_trace(trace_batch_size, row=4, col=1)
    data_fig.add_trace(trace_lr, row=4, col=1)
    data_fig.add_trace(trace_layers, row=4, col=1)
    data_fig.add_trace(trace_dropout, row=4, col=1)
    data_fig.add_trace(trace_hidden_size, row=4, col=1)

    # Display the mean square error values and optimization algorithms from all the permutations
    print("MSE from all studies:", mse_list)
    print("Optimizers from all studies:", optimizers)

    # Title overall container of subplots and x-axes of those subplots
    data_fig.update_layout(height=1200, width=1200, title="Hyperparameter Trends with High Features, "
                                                          "Medium Seasonality")
    data_fig.update_xaxes(title_text="Epochs", row=1, col=1)
    data_fig.update_xaxes(title_text="Input Time Steps", row=2, col=1)
    data_fig.update_xaxes(title_text="Output Time Steps", row=3, col=1)
    data_fig.update_xaxes(title_text="Trials", row=4, col=1)

    # Save the subplot container in an html file
    plot(data_fig, filename="experiment/results.html")
