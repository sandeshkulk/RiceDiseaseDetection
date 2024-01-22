from utils.model_creator import save_model_checkpoint

def train(model, X_train, y_train, batch_size, epochs, X_test, y_test):
    """
    Train a deep learning model on the given training data and save checkpoints.

    Parameters:
    - model: The deep learning model to be trained.
    - X_train (numpy.ndarray): Input features for the training set.
    - y_train (numpy.ndarray): Target labels for the training set.
    - batch_size (int): Number of samples per gradient update during training.
    - epochs (int): Number of epochs to train the model.
    - X_test (numpy.ndarray): Input features for the validation set.
    - y_test (numpy.ndarray): Target labels for the validation set.

    Returns:
    None

    This function trains the specified deep learning model on the provided training data,
    saves checkpoints during training, and prints a completion message.

    Example:
    >>> train(my_model, X_train_data, y_train_data, batch_size=32, epochs=10, X_val_data, y_val_data)
    """

    training_model = model.fit(X_train, y_train, batch_size=batch_size,
                               epochs=epochs,
                               validation_data=(X_test, y_test),
                               verbose=2,
                               shuffle=True
                               )

    save_model_checkpoint(tf_model=model, epoch=epochs)

    print("====================================================================================================")
    print("Training of the model on the dataset is completed")
    print("====================================================================================================")