import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series , format="-", start=0, end=None):
    '''
    series = [(serie_1, label_1), (serie_2, label_2), ...]
    '''
    
    plt.figure(figsize=(10, 6))
    for s, l in series:
        plt.plot(time[start:end], s[start:end], format, label= l)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend(loc='upper left')
    
    return plt
    
    
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    return plt

    
def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps


def plot_prediction(plot_data, title, delta=0):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    plt.figure(figsize=(10, 6))
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                   label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt