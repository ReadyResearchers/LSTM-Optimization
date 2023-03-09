import pickle
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objs as go
import os
import sys

if __name__ == "__main__":

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Controls: 10 epochs, 10 time steps, 10 projection size, 50 trials

    epochs = [5, 8, 10, 20, 30]
    time_steps = [3, 8, 10, 15, 20]
    projection_size = [5, 8, 10, 15, 20]
    trials = [10, 25, 50, 75, 100]

    mse_list = []
    optimizers = []

    data_fig = make_subplots(rows=4, cols=1)

    y_batch_size = []
    y_lr = []
    y_layers = []
    y_dropout = []
    y_hidden_size = []

    trace_batch_size = go.Scatter(x=epochs, y=y_batch_size, name="Batch Size", mode="lines", line=dict(color="blue"))
    trace_lr = go.Scatter(x=epochs, y=y_lr, name="Learning Rate", mode="lines", line=dict(color="red"))
    trace_layers = go.Scatter(x=epochs, y=y_layers, name="Number of Layers", mode="lines", line=dict(color="green"))
    trace_dropout = go.Scatter(x=epochs, y=y_dropout, name="Dropout Rate", mode="lines", line=dict(color="orange"))
    trace_hidden_size = go.Scatter(x=epochs, y=y_hidden_size, name="Hidden Size", mode="lines", line=dict(color="black"))

    for e in epochs:
        program = "hyper_param/lstm.py"
        csv = "data/city_temperature_compressed.csv"
        input = "[Year,Month,Day]"
        output = "[AvgTemperature]"
        control_time_steps = 10
        control_projection_size = 10
        control_trials = 50
        test_epochs = e
        command = f"python {program} {csv} {input} {output} {control_time_steps} {control_projection_size}" \
                  f" {control_trials} {test_epochs}"
        print(command)
        os.system(command)

        with open("experiment/best_values.pickle", "rb") as f:
            best_trial = pickle.load(f)

        mse_list.append(best_trial.value)
        optimizers.append(best_trial.params["optimizer"])

        y_batch_size.append((best_trial.params["batch_size"] - 10) / (100 - 10))
        y_lr.append((best_trial.params["lr"] - 0.00001) / (0.01 - 0.00001))
        y_layers.append((best_trial.params["n_layers"] - 2) / (5 - 2))
        y_dropout.append((best_trial.params["dropout"] - 0.1) / (0.5 - 0.1))
        y_hidden_size.append((best_trial.params["hidden_size"] - control_projection_size - 1) /
                             (2048 - control_projection_size - 1))

    trace_batch_size.update(y=y_batch_size)
    trace_lr.update(y=y_lr)
    trace_layers.update(y=y_layers)
    trace_dropout.update(y=y_dropout)
    trace_hidden_size.update(y=y_hidden_size)

    data_fig.add_trace(trace_batch_size, row=1, col=1)
    data_fig.add_trace(trace_lr, row=1, col=1)
    data_fig.add_trace(trace_layers, row=1, col=1)
    data_fig.add_trace(trace_dropout, row=1, col=1)
    data_fig.add_trace(trace_hidden_size, row=1, col=1)

    y_batch_size = []
    y_lr = []
    y_layers = []
    y_dropout = []
    y_hidden_size = []

    for t in time_steps:
        program = "hyper_param/lstm.py"
        csv = "data/city_temperature_compressed.csv"
        input = "[Year,Month,Day]"
        output = "[AvgTemperature]"
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

    for p in projection_size:
        program = "hyper_param/lstm.py"
        csv = "data/city_temperature_compressed.csv"
        input = "[Year,Month,Day]"
        output = "[AvgTemperature]"
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

    for t in trials:
        program = "hyper_param/lstm.py"
        csv = "data/city_temperature_compressed.csv"
        input = "[Year,Month,Day]"
        output = "[AvgTemperature]"
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


    print("MSE from all studies:", mse_list)
    print("Optimizers from all studies:", optimizers)

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

    data_fig.update_layout(height=1200, width=1200, title="Hyperparameter Trends with Low Features, Medium Seasonality")
    data_fig.update_xaxes(title_text="Epochs", row=1, col=1)
    data_fig.update_xaxes(title_text="Input Time Steps", row=2, col=1)
    data_fig.update_xaxes(title_text="Output Time Steps", row=3, col=1)
    data_fig.update_xaxes(title_text="Trials", row=4, col=1)

    plot(data_fig, filename="experiment/results.html")
