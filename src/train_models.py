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

def train_augmented_standard(x_train, y_train, x_val, y_val, x_test, y_test, run):
    """Construct, train and evaluate the standard network on augmented data.
    
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
    run_logdir = os.path.join(os.curdir, f"../../lr_logs/standard_aug_run{run}")
    
    # Create utility callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    save = keras.callbacks.ModelCheckpoint(f"../../models/aug_standard_model{run}.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    # Train model
    train_log = std_cnn.fit(x_train, y_train, epochs=100,
                   validation_data=(x_val, y_val),
                   callbacks=[save, tensorboard_cb, early_stop])

    best_model = keras.models.load_model(f"../../models/aug_standard_model{run}.h5")
    return best_model.evaluate(x_test, y_test)

def train_aux(x_train, y_train, x_val, y_val, x_test, y_test,  run):
    """Construct, train and evaluate the auxiliary network.
    Parameters
    ----------
    x_train : ndarray
        The training data.
    y_train : ndarray
        The feature vectors corresponding to the training data.
    x_val : ndarray
        The validation data.
    y_val : ndarray
        The feature vectors corresponding to the validation data.
    x_test : ndarray
        The test data.
    y_test : ndarray
        The feature vectors corresponding to the test data.
    run : integer
        Indicates which training run this is.
    
    Returns
    -------
    tuple
        A tuple containing the MSE and loss of the final network when evaluated on the test set.
    """
    probe_dir('../../lr_logs/')
    probe_dir('../../models/')
    aux_cnn = construct_aux()
    
    #Tensorboard setup
    run_logdir = os.path.join(os.curdir, f"../../lr_logs/aux_run{run}")
    
    # Create utility callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    save = keras.callbacks.ModelCheckpoint(f"../../models/aux_model{run}.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    y_train_bent = y_train[:, 0]
    y_train_fr = y_train[:, 1]
    y_train_cores = y_train[:, 2]
    y_train_size = y_train[:, 3]
    train_labels = {
        'bent_out': y_train_bent,
        'fr_out': y_train_fr,
        'cores_out': y_train_cores,
        'size_out': y_train_size
    }
    
    y_val_bent = y_val[:, 0]
    y_val_fr = y_val[:, 1]
    y_val_cores = y_val[:, 2]
    y_val_size = y_val[:, 3]
    val_labels = {
        'bent_out': y_val_bent,
        'fr_out': y_val_fr,
        'cores_out': y_val_cores,
        'size_out': y_val_size
    }
    
    y_test_bent = y_test[:, 0]
    y_test_fr = y_test[:, 1]
    y_test_cores = y_test[:, 2]
    y_test_size = y_test[:, 3]
    test_labels = {
        'bent_out': y_test_bent,
        'fr_out': y_test_fr,
        'cores_out': y_test_cores,
        'size_out': y_test_size
    }
    
    # Train model
    train_log = aux_cnn.fit(x_train, train_labels, epochs=100,
                   validation_data=(x_val, val_labels),
                   callbacks=[save, tensorboard_cb, early_stop])

    best_model = keras.models.load_model(f"../../models/aux_model{run}.h5")
    return best_model.evaluate(x_test, test_labels)

def construct_aux():
    """Construct the auxiliary network.
    
    Returns
    -------
    Model
        The compiled TensorFlow model representing the network.
    """

    #Convolutional layer wrapper
    base_conv = partial(keras.layers.Conv2D, kernel_size=7, activation='relu', padding='same')
    
    cnn_input = keras.layers.Input(shape=[150, 150, 1])
    conv0 = base_conv(filters=64, kernel_size=14)(cnn_input)
    pool0 = keras.layers.MaxPooling2D(2)(conv0)
    
    #First convolutional block
    conv1 = base_conv(filters=128)(pool0)
    conv2 = base_conv(filters=128)(conv1)
    pool1 = keras.layers.MaxPooling2D(2)(conv2)
    #Second convolutional block
    conv3 = base_conv(filters=256)(pool1)
    conv4 = base_conv(filters=256)(conv3)
    pool2 = keras.layers.MaxPooling2D(2)(conv4)
    
    #Set up for fully connected blocks
    flatten = keras.layers.Flatten()(pool2)

    # First dense layer
    dense0 = keras.layers.Dense(150, activation='elu', kernel_initializer="he_normal")(flatten)
    drop0 = keras.layers.Dropout(0.5)(dense0)
    
    # Second dense layer
    dense1 = keras.layers.Dense(75, activation='elu', kernel_initializer="he_normal")(drop0)
    drop1 = keras.layers.Dropout(0.5)(dense1)
    
    # Feature outputs
    bent_out = keras.layers.Dense(units=1, activation='sigmoid', name="bent_out")(drop1)
    fr_out = keras.layers.Dense(units=1, activation='sigmoid', name="fr_out")(drop1)
    cores_out = keras.layers.Dense(units=1, activation='sigmoid', name="cores_out")(drop1)
    size_out = keras.layers.Dense(units=1, activation='sigmoid', name="size_out")(drop1)
    
    aux_cnn = keras.Model(inputs = cnn_input, outputs = [bent_out, fr_out, cores_out, size_out])
    
    optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
    loss = {
        'bent_out': 'binary_crossentropy',
        'fr_out': 'mean_squared_error',
        'cores_out': 'mean_squared_error',
        'size_out': 'mean_squared_error'
    }
    metrics = {
        'bent_out': 'accuracy',
        'fr_out': 'mean_absolute_error',
        'cores_out': 'mean_absolute_error',
        'size_out': 'mean_absolute_error'
    }
    aux_cnn.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return aux_cnn

def train_aux_man(x_train, y_train, x_val, y_val, x_test, y_test,  run):
    """Construct, train and evaluate the auxiliary network.
    This function is intended for use with the auxiliary features that have manually extracted labels added to them.
    Parameters
    ----------
    x_train : ndarray
        The training data.
    y_train : ndarray
        The feature vectors corresponding to the training data.
    x_val : ndarray
        The validation data.
    y_val : ndarray
        The feature vectors corresponding to the validation data.
    x_test : ndarray
        The test data.
    y_test : ndarray
        The feature vectors corresponding to the test data.
    run : integer
        Indicates which training run this is.
    
    Returns
    -------
    tuple
        A tuple containing the MSE and loss of the final network when evaluated on the test set.
    """
    probe_dir('../../lr_logs/')
    probe_dir('../../models/')
    aux_cnn = construct_aux()
    
    #Tensorboard setup
    run_logdir = os.path.join(os.curdir, f"../../lr_logs/man_aux_run{run}")
    
    # Create utility callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    save = keras.callbacks.ModelCheckpoint(f"../../models/man_aux_model{run}.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    y_train_bent = y_train[:, 0]
    y_train_fr = y_train[:, 1]
    y_train_cores = y_train[:, 2]
    y_train_size = y_train[:, 3]
    train_labels = {
        'bent_out': y_train_bent,
        'fr_out': y_train_fr,
        'cores_out': y_train_cores,
        'size_out': y_train_size
    }
    
    y_val_bent = y_val[:, 0]
    y_val_fr = y_val[:, 1]
    y_val_cores = y_val[:, 2]
    y_val_size = y_val[:, 3]
    val_labels = {
        'bent_out': y_val_bent,
        'fr_out': y_val_fr,
        'cores_out': y_val_cores,
        'size_out': y_val_size
    }
    
    y_test_bent = y_test[:, 0]
    y_test_fr = y_test[:, 1]
    y_test_cores = y_test[:, 2]
    y_test_size = y_test[:, 3]
    test_labels = {
        'bent_out': y_test_bent,
        'fr_out': y_test_fr,
        'cores_out': y_test_cores,
        'size_out': y_test_size
    }
    
    # Train model
    train_log = aux_cnn.fit(x_train, train_labels, epochs=100,
                   validation_data=(x_val, val_labels),
                   callbacks=[save, tensorboard_cb, early_stop])

    best_model = keras.models.load_model(f"../../models/man_aux_model{run}.h5")
    return best_model.evaluate(x_test, test_labels)

def train_wide(x_train_aux, x_train, y_train, x_val_aux, x_val, y_val, x_test_aux, x_test, y_test,  run):
    """Construct, train and evaluate the wide network.
    Parameters
    ----------
    x_train_aux : ndarray
        The auxiliary engineered features for the training data.
    x_train : ndarray
        The training data.
    y_train : ndarray
        The class labels of the training data.
    x_val_aux : ndarray
        The auxiliary engineered features for the validation data.
    x_val : ndarray
        The validation data.
    y_val : ndarray
        The class labels of the validation data.
    x_test_aux : ndarray
        The auxiliary engineered features for the test data.
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
    wide = construct_wide()

    #Tensorboard setup
    run_logdir = os.path.join(os.curdir, f"../../lr_logs/wide_run{run}")
    # Create utility callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    save = keras.callbacks.ModelCheckpoint(f"../../models/wide_model{run}.h5", save_best_only=True)
    tensorboard = keras.callbacks.TensorBoard(run_logdir)

    # Train network
    train_log = wide.fit((x_train_aux, x_train), y_train, epochs=100,
                    validation_data=((x_val_aux, x_val), y_val),
                    callbacks=[early_stop, save, tensorboard])
    
    best_wide = keras.models.load_model(f"../../models/wide_model{run}.h5")
    return best_wide.evaluate((x_test_aux, x_test), y_test)

def construct_wide():
    """Construct the wide network.
    
    Returns
    -------
    Model
        The compiled TensorFlow model representing the network.
    """

    #Create wrapper for my convolutional layer
    base_conv = partial(keras.layers.Conv2D, kernel_size=7, activation='relu', padding='same')
    
    #Define network
    main_input = keras.layers.Input(shape=[150, 150, 1])
    aux_input = keras.layers.Input(shape=[4])
    conv0 = keras.layers.Conv2D(filters=64, kernel_size=14)(main_input)
    pool1 = keras.layers.MaxPooling2D(2)(conv0)
    conv1 = base_conv(filters=128)(pool1)
    conv2 = base_conv(filters=128)(conv1)
    pool2 = keras.layers.MaxPooling2D(2)(conv2)
    conv3 = base_conv(filters=256)(pool2)
    conv4 = base_conv(filters=256)(conv3)
    pool3 = keras.layers.MaxPooling2D(2)(conv4)
    flatten = keras.layers.Flatten()(pool3)
    concat = keras.layers.concatenate([aux_input, flatten])
    dense1 = keras.layers.Dense(units=150, activation='elu', kernel_initializer="he_normal")(concat)
    drop1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(units=75, activation='elu', kernel_initializer="he_normal")(drop1)
    drop2 = keras.layers.Dropout(0.5)(dense2)
    output = keras.layers.Dense(units=4, activation='softmax')(drop2)
    wide = keras.Model(inputs=[aux_input, main_input], outputs=[output])

    # Compile network
    optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
    wide.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=[keras.metrics.CategoricalAccuracy()])
    
    return wide

def train_wide_man(x_train_aux, x_train, y_train, x_val_aux, x_val, y_val, x_test_aux, x_test, y_test,  run):
    """Construct, train and evaluate the wide network.
    This function is intended for use with the auxiliary features that have manually extracted labels added to them.
    Parameters
    ----------
    x_train_aux : ndarray
        The auxiliary engineered features for the training data.
    x_train : ndarray
        The training data.
    y_train : ndarray
        The class labels of the training data.
    x_val_aux : ndarray
        The auxiliary engineered features for the validation data.
    x_val : ndarray
        The validation data.
    y_val : ndarray
        The class labels of the validation data.
    x_test_aux : ndarray
        The auxiliary engineered features for the test data.
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
    wide = construct_wide()

    #Tensorboard setup
    run_logdir = os.path.join(os.curdir, f"../../lr_logs/man_wide_run{run}")
    # Create utility callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    save = keras.callbacks.ModelCheckpoint(f"../../models/man_wide_model{run}.h5", save_best_only=True)
    tensorboard = keras.callbacks.TensorBoard(run_logdir)

    # Train network
    train_log = wide.fit((x_train_aux, x_train), y_train, epochs=100,
                    validation_data=((x_val_aux, x_val), y_val),
                    callbacks=[early_stop, save, tensorboard])
    
    best_wide = keras.models.load_model(f"../../models/man_wide_model{run}.h5")
    return best_wide.evaluate((x_test_aux, x_test), y_test)