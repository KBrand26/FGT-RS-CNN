from functools import partial
from tensorflow import keras
import os
import tensorflow_addons as tfa

def probe_dir(dir_path):
    """Check whether directory exists and if not create it

    Args:
        dir_path (String): Path to the required directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train_standard(x_train, y_train, x_val, y_val, x_test, y_test, run):
    """Construct, train and evaluate the standard network.
    Parameters
    ----------
    x_train : ndarray
        The training data.
    y_train : ndarray
        The class labels of the training data.
    x_val : ndarray
        The validation data.
    y_val : ndarray
        The class labels of the validation data.
    x_test : ndarray
        The test data.
    y_test : ndarray
        The class labels of the test data.
    run : integer
        Indicates which training run this is.
    
    Returns
    -------
    tuple
        A tuple containing the accuracy and loss of the final network when evaluated on the test set.
    """
    probe_dir('../../lr_logs/')
    probe_dir('../../models/')
    std_cnn = construct_standard()
    
    #Tensorboard setup
    run_logdir = os.path.join(os.curdir, f"../../lr_logs/standard_run{run}")
    
    # Create utility callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    save = keras.callbacks.ModelCheckpoint(f"../../models/standard_model{run}.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    # Train model
    train_log = std_cnn.fit(x_train, y_train, epochs=100,
                   validation_data=(x_val, y_val),
                   callbacks=[save, tensorboard_cb, early_stop])

    best_model = keras.models.load_model(f"../../models/standard_model{run}.h5")
    return best_model.evaluate(x_test, y_test)

def construct_standard():
    """Construct the standard network.
    
    Returns
    -------
    Model
        The compiled TensorFlow model representing the network.
    """

    #Convolutional layer wrapper
    base_conv = partial(keras.layers.Conv2D, kernel_size=7, activation='relu', padding='same')
    
    std_cnn = keras.models.Sequential()
    filters = 64
    std_cnn.add(base_conv(filters=filters, kernel_size=14, input_shape=[150, 150, 1]))
    for i in range(2):
        filters <<= 1
        std_cnn.add(keras.layers.MaxPooling2D(2))
        std_cnn.add(base_conv(filters=filters))
        std_cnn.add(base_conv(filters=filters))
    std_cnn.add(keras.layers.MaxPooling2D(2))
    std_cnn.add(keras.layers.Flatten())
    std_cnn.add(keras.layers.Dense(units=150, activation='elu', kernel_initializer="he_normal"))
    std_cnn.add(keras.layers.Dropout(0.5))
    std_cnn.add(keras.layers.Dense(units=75, activation='elu', kernel_initializer="he_normal"))
    std_cnn.add(keras.layers.Dropout(0.5))
    std_cnn.add(keras.layers.Dense(units=4, activation='softmax'))
    optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
    std_cnn.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=[keras.metrics.CategoricalAccuracy()])
    return std_cnn

def train_derotated_standard(x_train, y_train, x_val, y_val, x_test, y_test, run):
    """Construct, train and evaluate the standard network with derotated images.
    Parameters
    ----------
    x_train : ndarray
        The training data.
    y_train : ndarray
        The class labels of the training data.
    x_val : ndarray
        The validation data.
    y_val : ndarray
        The class labels of the validation data.
    x_test : ndarray
        The test data.
    y_test : ndarray
        The class labels of the test data.
    run : integer
        Indicates which training run this is.
    
    Returns
    -------
    tuple
        A tuple containing the accuracy and loss of the final network when evaluated on the test set.
    """
    probe_dir('../../lr_logs/')
    probe_dir('../../models/')
    std_cnn = construct_standard()
    
    #Tensorboard setup
    run_logdir = os.path.join(os.curdir, f"../../lr_logs/standard_derotated_run{run}")
    
    # Create utility callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    save = keras.callbacks.ModelCheckpoint(f"../../models/derotated_standard_model{run}.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    # Train model
    train_log = std_cnn.fit(x_train, y_train, epochs=100,
                   validation_data=(x_val, y_val),
                   callbacks=[save, tensorboard_cb, early_stop])

    best_model = keras.models.load_model(f"../../models/derotated_standard_model{run}.h5")
    return best_model.evaluate(x_test, y_test)