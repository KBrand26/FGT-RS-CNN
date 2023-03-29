# FGT-RS-CNN
This project consists of training various neural networks for the the morphological classification of radio galaxies. In this repository we investigate possible benefits of standardising rotations and guiding neural networks to look for specific features. Most of these networks were built in TensorFlow, but the XOR networks were built from scratch in Python.

# How to prepare the data
1. Download dataset from https://doi.org/10.5281/zenodo.7645530 and unpack into FITS directory.
2. Execute the following command to process the data in preparation for experimentation:
```
python src/data_prep.py -d
```

# How to experiment on XOR data
The **src/XOR_Networks/eval.py** file is used to train a standard and guided XOR network and to generate the required plots for performance evaluation.

To generate violin plots of the epochs it took to train the networks run the following command in the **src/XOR_Networks** directory:
```
python eval.py
```

To generate loss curves for the training of the XOR networks run the following command in the **src/XOR_Networks** directory:
```
python eval.py -l
```

# How to access Jupyter notebooks
Most of the networks were constructed, trained and evaluated using TensorFlow. To make this process as efficient as possible, I made use of a Docker container that allows TensorFlow to make use of my GPU. To launch this Docker container and access the notebooks run  the following commands in order:

```
docker run --gpus all -it -v "$PWD":/tf/notebooks -p 8001:8001 -p 8002:8002 --rm tensorflow/tensorflow:latest-gpu-jupyter bash

cd notebooks

pip install -r requirements.txt

jupyter notebook --ip 0.0.0.0 --port 8001 --no-browser --allow-root
```

# How to conduct experiments for FRGMRC data
The experiments for the various network architectures have been separated into their own notebooks in the **src/Galaxy_Networks/** directory. You should run all of these notebooks to generate the necessary training logs that are used to evaluate the performance of the networks. Please note that even with a GPU, running all of these notebooks will take quite long.

Once the notebooks have been executed, the **gen_galaxy_violin_plots.ipynb** notebook can be executed to generate violin plots that compare the performance of the various architectures.

# Project layout
**plots/** : The plots directory contains the plots that were produced for reporting results.

**models/** : Trained models representing the various networks will be stored within this directory.

**lr_logs/**: This directory contains the Tensorboard logs for the training of the various neural networks.

**src/** : All Python scripts and Jupyter notebooks can be found within the src directory.

**src/XOR_Networks/** : The XOR_Networks directory contains the Python files that were used to train all of the networks for the XOR binary operator. The _layer.py_ file is a class that represents layers within the networks. The _eval.py_ file is used to record certain performance metrics during training of the XOR networks and constructs plots for them. The rest of the file names are self explanatory.

**src/Galaxy_Networks/** : This directory contains the notebooks that were used to train the various neural network architectures on the FRGMRC data.

**src/custom_plots.py** : This Python file contains utility methods used to create custom plots for the various networks.

**src/data_prep.py** : This Python file contains utility methods used to prepare the data for experimentation.

**src/train_models.py** : This Python file contains functions that are used to construct, train and evaluate the various networks.

**src/gen_galaxy_violin_plots.ipynb** : This Jupyter notebook is used to generate violin plots for the FRGMRC networks.
