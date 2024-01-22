import tensorflow as tf
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def loadData(input_shape_2D, seed):
    """
    Load image data from the specified directory using TensorFlow's image_dataset_from_directory.

    Parameters:
    - input_shape_2D (tuple): The desired shape of input images in 2D (height, width).
    - seed (int): Seed for reproducibility when shuffling the dataset.

    Returns:
    - data (tf.data.Dataset): A TensorFlow dataset containing the loaded images and labels.

    This function uses TensorFlow's image_dataset_from_directory to load image data from the
    'rice_disease' directory. It returns a TensorFlow dataset object containing images and labels.

    Example:
    >>> input_shape = (224, 224)
    >>> seed_value = 42
    >>> dataset = loadData(input_shape, seed_value)
    """

    data = tf.keras.utils.image_dataset_from_directory(directory="rice_disease",
                                                       labels='inferred',
                                                       label_mode='int',
                                                       class_names=None,
                                                       color_mode='rgb',
                                                       image_size=input_shape_2D,
                                                       seed=seed)
    return data