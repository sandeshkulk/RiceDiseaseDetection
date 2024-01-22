from keras.utils import to_categorical

def oneHotEncoder(class_names, y_train, y_test):
    """
    Convert class labels to one-hot encoded format using Keras' to_categorical.

    Parameters:
    - class_names (list): List of class labels representing different categories.
    - y_train (numpy.ndarray): Target labels for the training set.
    - y_test (numpy.ndarray): Target labels for the test set.

    Returns:
    - y_train_encoded (numpy.ndarray): One-hot encoded labels for the training set.
    - y_test_encoded (numpy.ndarray): One-hot encoded labels for the test set.

    This function uses Keras' to_categorical to convert integer class labels to
    one-hot encoded format for both training and test sets.

    Example:
    >>> classes = ['class_A', 'class_B', 'class_C']
    >>> y_train_data, y_test_data = oneHotEncoder(classes, y_train, y_test)
    """

    y_train_encoded = to_categorical(y_train, len(class_names))
    y_test_encoded = to_categorical(y_test, len(class_names))
    return y_train_encoded, y_test_encoded