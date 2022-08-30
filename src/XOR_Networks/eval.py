from merged_network import MergedNetwork
from std_network import StandardNetwork
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt

def evaluate_networks():
    """This function is used to train standard neural networks
    as well as merged neural networks for application to the XOR daataset.
    
    The number of epochs it took for the network to reach a preselected loss during each training run
    is recorded and presented in box plots.
    """
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_std = np.array([0, 1, 1, 0])
    y_merged = np.array([[0, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1]])
    epochs_std = []
    epochs_merged1 = []
    epochs_merged2 = []
    epochs_merged3 = []
    epochs_merged4 = []

    # Train each of the networks 20 times
    for i in range(20):
        print(f'Starting training run {i}')
        sn = StandardNetwork(0.1, True)
        mn1 = MergedNetwork(0.1, np.array([0.34, 0.33, 0.33]), True)
        mn2 = MergedNetwork(0.1, np.array([0.5, 0.25, 0.25]), True)
        mn3 = MergedNetwork(0.1, np.array([0.25, 0.5, 0.25]), True)
        mn4 = MergedNetwork(0.1, np.array([0.25, 0.25, 0.5]), True)

        tmp_loss = sn.train(X, y_std, 10000)
        epochs_std.append(len(tmp_loss))

        tmp_loss = mn1.train(X, y_merged, 10000)
        epochs_merged1.append(len(tmp_loss))

        tmp_loss = mn2.train(X, y_merged, 10000)
        epochs_merged2.append(len(tmp_loss))

        tmp_loss = mn3.train(X, y_merged, 10000)
        epochs_merged3.append(len(tmp_loss))

        tmp_loss = mn4.train(X, y_merged, 10000)
        epochs_merged4.append(len(tmp_loss))

    epochs_std = np.array(epochs_std)
    epochs_merged1 = np.array(epochs_merged1)
    epochs_merged2 = np.array(epochs_merged2)
    epochs_merged3 = np.array(epochs_merged3)
    epochs_merged4 = np.array(epochs_merged4)

    # Create the boxplots
    epoch_results = {"Standard": epochs_std, "Merged\nEven": epochs_merged1, "Merged\nXOR": epochs_merged2,
                "Merged\nNAND": epochs_merged3, "Merged\nOR": epochs_merged4}
    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.boxplot(epoch_results.values())
    ax.set_xticklabels(epoch_results.keys())
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel('Epochs taken', labelpad=6, fontsize=14)
    ax.set_xlabel('Network trained', labelpad=10, fontsize=14)
    if not os.path.exists('../../plots/XOR/'):
        os.makedirs('../../plots/XOR/')
    plt.savefig('../../plots/XOR/XOR_boxplot.eps', format='eps', bbox_inches="tight", pad_inches=0)
    plt.close()

def gen_loss_plots():
    """
    This function trains standard and merged XOR neural networks over twenty training runs.
    The loss values are recorded after each epoch and are used to plot the loss curves
    for all of the training runs.
    """
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_std = np.array([0, 1, 1, 0])
    y_merged = np.array([[0, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1]])
    epochs = np.array([i for i in range(1, 10001)]*20)
    runs = []
    for r in range(1, 21):
        tmp = [r for i in range(10000)]
        runs += tmp
    runs = np.array(runs)
    losses_std = []
    losses_m1 = []
    losses_m2 = []
    losses_m3 = []
    losses_m4 = []

    # Perform twenty training runs of each network
    for i in range(20):
        print(f'Starting training run {i}')
        sn = StandardNetwork(0.1, False)
        mn1 = MergedNetwork(0.1, np.array([0.34, 0.33, 0.33]), False)
        mn2 = MergedNetwork(0.1, np.array([0.5, 0.25, 0.25]), False)
        mn3 = MergedNetwork(0.1, np.array([0.25, 0.5, 0.25]), False)
        mn4 = MergedNetwork(0.1, np.array([0.25, 0.25, 0.5]), False)

        tmp_loss = sn.train(X, y_std, 10000)
        losses_std += tmp_loss.tolist()

        tmp_loss = mn1.train(X, y_merged, 10000)
        losses_m1 += tmp_loss.tolist()

        tmp_loss = mn2.train(X, y_merged, 10000)
        losses_m2 += tmp_loss.tolist()

        tmp_loss = mn3.train(X, y_merged, 10000)
        losses_m3 += tmp_loss.tolist()

        tmp_loss = mn4.train(X, y_merged, 10000)
        losses_m4 += tmp_loss.tolist()

    # Convert the loss values to dataframes
    losses_std = np.array(losses_std)
    losses_m1 = np.array(losses_m1)
    losses_m2 = np.array(losses_m2)
    losses_m3 = np.array(losses_m3)
    losses_m4 = np.array(losses_m4)
    data_std = {'Epoch': epochs, 'Run': runs, 'Loss': losses_std}
    df_std = pd.DataFrame(data_std)
    data_m1 = {'Epoch': epochs, 'Run': runs, 'Loss': losses_m1}
    df_m1 = pd.DataFrame(data_m1)
    data_m2 = {'Epoch': epochs, 'Run': runs, 'Loss': losses_m2}
    df_m2 = pd.DataFrame(data_m2)
    data_m3 = {'Epoch': epochs, 'Run': runs, 'Loss': losses_m3}
    df_m3 = pd.DataFrame(data_m3)
    data_m4 = {'Epoch': epochs, 'Run': runs, 'Loss': losses_m4}
    df_m4 = pd.DataFrame(data_m4)

    # Plot the loss curves using Seaborn
    print('Plotting')
    sns.lineplot(data=df_std, x='Epoch', y='Loss')
    print(1)
    sns.lineplot(data=df_m1, x='Epoch', y='Loss')
    print(2)
    sns.lineplot(data=df_m2, x='Epoch', y='Loss')
    print(3)
    sns.lineplot(data=df_m3, x='Epoch', y='Loss')
    print(4)
    sns.lineplot(data=df_m4, x='Epoch', y='Loss')
    plt.plot([i for i in range(1, 10001)], [0.01 for j in range(1, 10001)])
    plt.xlabel('Epoch', labelpad=10, fontsize=13)
    plt.ylabel('Loss', labelpad=6, fontsize=13)
    plt.legend(['Standard network', 'Merged Even', 'Merged XOR', 'Merged NAND', 'Merged OR', 'Early stopping threshold'], prop={'size': 13})
    if not os.path.exists('../../plots/XOR/'):
        os.makedirs('../../plots/XOR/')
    plt.savefig('../../plots/XOR/XOR_sns_loss.eps', format='eps', bbox_inches="tight", pad_inches=0)
    plt.close()
    
if __name__ == '__main__':
    gen_loss_plots()