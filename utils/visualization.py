import numpy as np
import matplotlib.pyplot as plt

def show_plot(X_test, y_prediction, y_test, leaf_class):
    """
    Display a grid of images from the test set along with predicted and ground truth labels.

    Parameters:
    - X_test (numpy.ndarray): The input images from the test set.
    - y_prediction (numpy.ndarray): Predicted labels for the corresponding images in X_test.
    - y_test (numpy.ndarray): Ground truth labels for the test set.
    - leaf_class (list): List of class labels representing different types of leaves.

    Returns:
    None

    This function generates a grid of images along with their predicted and ground truth labels,
    highlighting correct predictions in blue and incorrect predictions in red.

    Example:
    >>> show_plot(X_test, y_prediction, y_test, leaf_class)
    """

    fig = plt.figure(figsize=(25, 25))
    for i, idx in enumerate(np.random.choice(X_test.shape[0], size=32, replace=True)):
        ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(X_test[idx]))

        pred_idx = np.argmax(y_prediction[idx])
        confidence = y_prediction[idx][pred_idx] * 100

        true_idx = np.argmax(y_test[idx])

        ax.set_title("{} ({:.2f}%)\nGround Truth: {}".format(
            leaf_class[pred_idx], confidence, leaf_class[true_idx]),
            color=("blue" if pred_idx == true_idx else "red"))

    plt.tight_layout()
    plt.show()