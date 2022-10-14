from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os.path import exists
from os import getcwd, mkdir

def extract_class_pred(preds):
    """This function is used to extract the desired predictions from the outputs of a trained model

    Args:
        preds (ndarray): The matrix of predictions made by the model

    Returns:
        ndarray: The desired predictions for model evaluation 
    """
    return preds[0]

def tensor_to_val(tensor):
    """Converts a given tensor to a ndarray

    Args:
        tensor (tensor): The tensor that needs to be converted.

    Returns:
        ndarray: The ndarray corresponding to the given tensor.
    """
    return tf.make_ndarray(tensor)

def gen_violin_plot(df, x, y, hue, file, split=True, only_data=False, limit=False, limit_range=[]):
    """Generates a violin plot for the given data

    Args:
        df (dataframe): The dataframe containing the data that should be used.
        x (String): The name of the column in the dataframe that should be used for the x axis.
        y (String): The name of the column in the dataframe that should be used for the y axis.
        hue (String): The name of the column in the dataframe to use to determine the hue.
        file (String): The name to use when saving the violin plot to a file.
        split (bool, optional): A flag that indicates whether the violin plot should be split into two halves
                                to indicate differences for a binary variable. Defaults to True.
        only_data (bool, optional): A flag that indicates whether the violin plot should only use estimations for the
                                    range of values observed in the given data. Defaults to False.
        limit (bool, optional): A flag that indicates whether the y-axis should be limited. Defaults to False.
        limit_range (list, optional): The range of values to which the y-axis should be limited. Defaults to [].
    """
    # Set figure size and text size
    sns.set(rc={'figure.figsize':(14,10)})
    sns.set(font_scale = 2)
    
    # Generate plot
    if split:
        if only_data:
            ax = sns.violinplot(x=x, y=y, hue=hue, data=df, linewidth=3, split=split, cut=0)
        else:
            ax = sns.violinplot(x=x, y=y, hue=hue, data=df, linewidth=3, split=split)
    else:
        if only_data:
            ax = sns.violinplot(x=x, y=y, data=df, linewidth=3, split=split, cut=0)        
        else:
            ax = sns.violinplot(x=x, y=y, data=df, linewidth=3, split=split)
            
    # Update axis limit if necessary
    if limit:
        ax.set_ylim(limit_range)
        
    # Add space between x ticks and label
    ax.set_xlabel('\nModel')
    
    if not exists("../plots/"):
        if getcwd().split("/")[-1] == "src":
            mkdir("../plots/")
        else:
            print("This function should only be called from the src directory")
            
    
    # Save plot
    location = '../plots/' + file
    plt.savefig(location, format='eps', bbox_inches="tight", pad_inches=0)
    plt.close()
    
def normalize_images(X):
    """Normalize the given images

    Args:
        X (ndarray): An array of images that need to be normalized

    Returns:
        ndarray: The normalized images
    """
    return np.array(list(map(normalize_image, X)))
    
def normalize_image(img):
    """Applies normalization to the given image. Ensures that pixels will have values between 0 and 1.

    Args:
        img (ndarray): The image that needs to be normalized.

    Returns:
        ndarray: The normalized image.
    """
    bot = np.min(img)
    top = np.max(img)
    norm = (img - bot)/(top - bot)
    return norm

def plot_logs(log_names, legend_labels, titles, acc_name, loss_name):
    """Generates plots showing the loss and accuracy curves from given training logs.
    Parameters
    ----------
    log_names : list
        List of the logs that plots have to be generated for.
    legend_labels : list
        The labels to use in the legend of the plots.
    titles : list
        The titles to use for the various plots.
    """
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    fig, ax = plt.subplots(1, 2)

    for name in log_names:
        ea = EventAccumulator(name)
        ea.Reload()
        ct, epoch_acc, acc = zip(*ea.Tensors(acc_name))
        ct = list(ct)
        epoch_acc = list(epoch_acc)
        acc = list(acc)
        acc = list(map(tensor_to_val, acc))
        ct, epoch_loss, loss = zip(*ea.Tensors(loss_name))
        ct = list(ct)
        epoch_loss = list(epoch_loss)
        loss = list(loss)
        loss= list(map(tensor_to_val, loss))

        ax[0].plot(epoch_acc, acc)
        ax[1].plot(epoch_loss, loss)
    ax[0].legend(legend_labels)
    ax[1].legend(legend_labels)
    ax[0].title.set_text(titles[0])
    ax[1].title.set_text(titles[1])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    plt.show()