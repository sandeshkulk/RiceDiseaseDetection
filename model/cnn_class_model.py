from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

def build_model(input_shape_3D):
    """
    Build and compile a convolutional neural network model for image classification.

    Parameters:
    - input_shape_3D (tuple): The desired shape of input images in 3D (height, width, channels).

    Returns:
    - model (keras.models.Sequential): The compiled convolutional neural network model.

    This function defines a convolutional neural network (CNN) model for image classification.
    The model architecture includes convolutional layers with max pooling, dropout layers,
    and fully connected layers. The model is compiled using the Adam optimizer and categorical
    crossentropy loss.

    Example:
    >>> input_shape = (224, 224, 3)
    >>> my_model = build_model(input_shape)
    """

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu', use_bias=False, input_shape=input_shape_3D))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu', use_bias=False))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu', use_bias=False))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu', use_bias=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=3, padding='same', strides=1, activation='relu', use_bias=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Define the model with optimizer, loss, metrics
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model