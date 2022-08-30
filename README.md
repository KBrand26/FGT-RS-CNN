# FGT-RS-CNN
This project consists of training various neural networks for the the morphological classification of radio galaxies. In this repository we investigate possible benefits of standardising rotations and guiding neural networks to look for specific features. Most of these networks were built in TensorFlow, but the XOR networks were built from scratch in Python.

## How to access Jupyter notebooks
As mentioned earlier, most of the networks were constructed, trained and evaluated using TensorFlow. To make this process as efficient as possible, I made use of a Docker container that allows TensorFlow to make use of my GPU. To launch this Docker container and access the notebooks run  the following commands in order:
```
docker run --gpus all -it -v "$PWD":/tf/notebooks -p 8001:8001 -p 8002:8002 --rm tensorflow/tensorflow:latest-gpu-jupyter bash

cd notebooks

pip install -r requirements.txt

jupyter notebook --ip 0.0.0.0 --port 8001 --no-browser --allow-root
```

## Project layout
**plots/** : The plots directory contains the plots that were produced for reporting results.

**models/** : Trained models representing the various networks will be stored within this directory.

**lr_logs/**: This directory contains the Tensorboard logs for the training of the various MNIST neural networks.

**src/** : All Python and Jupyter notebooks can be found within the src directory.

**src/XOR_Networks/** : The XOR_Networks directory contains the Python files that were used to train all of the networks for the XOR binary operator. The _layer.py_ file is a class that represents layers within the networks. The _eval.py_ file is used to record certain performance metrics during training of the XOR networks and constructs boxplots for them. The rest of the file names are self explanatory.

**src/Demo/**: This directory contains the Jupyter notebook that was used during the project demo.

**src/custom_plots.py** : This Python file contains utility methods used to create custom loss curves for the various networks.

**src/data_prep.py** : This Python file contains utility methods used to prepare the feature vectors for training samples.

**src/train_models.py** : This Python file contains functions that are used to construct, train and evaluate the various CIFAR10 and MNIST networks.

**src/Gen_Boxplots.ipynb** : This Jupyter notebook is used to generate boxplots for the MNIST and CIFAR10 networks.