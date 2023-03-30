import tensorflow as tf
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Python.config import Config

########################################################################
# Used to visualise data predictions
########################################################################

def load_colour_map(path:str) -> dict:
    """Loads colour map of the image segmentation masks.

    Args:
        path (str) : Path of colour map.

    Returns:
        colour_map (dict) : Dictionary of colour values for each class.
    """
    with open(path, "r") as file:
        colour_map = json.load(file)
    classes = colour_map.keys()
    values = [colour_map.get(key) for key in colour_map.keys()]
    colour_map = dict(zip(values, classes))
    return colour_map


def one_hot_to_categorical(one_hot: tf.Tensor) -> tf.Tensor:
    """Takes one-hot encoded data and changes it to categorical encoding.

    Args:
        one_hot (tf.Tensor) : Dataset tensor with one-hot encoded masks.

    Returns:
        categorical (tf.Tensor) : Dataset tensor with categorical 
            encoding.
    """
    get_categorical = lambda x : np.argmax(x)
    categorical = np.apply_along_axis(get_categorical, axis=-1, arr=one_hot)
    return categorical


def remove_axis_labels(axis: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    """Removes Axis labels.
    
    Args:
        axis (matplotlib.axes.Axes) : Axis on images.
    
    Returns:
        axis (matplotlib.axes.Axes) : Removed axis on images."""
    axis.get_yaxis().set_visible(False)
    axis.get_xaxis().set_visible(False)
    return axis


def compare_model_predictions(
    model: tf.keras.Model, image: np.ndarray, mask_true: np.ndarray
):
    """Prints original image, segmentation mask and prediction side to
        side with a colour bar.
        
        Args:
            model (tf.keras.Model) : Model used to create predictions.
            image (np.ndarray) : Original Image.
            mask_true (np.ndarray) : Segmentation mask.

            """
    dataset = tf.data.Dataset.from_tensors(image).batch(1)
    mask_predicted = model.predict(dataset)
    mask_predicted = np.squeeze(mask_predicted)
    mask_predicted = one_hot_to_categorical(mask_predicted)
    colour_map = load_colour_map(Config.colour_map_path)
    n_classes = len(colour_map.keys())
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1), cmap.N)
    fig = plt.figure(figsize=[16, 6])
    ax_image = fig.add_axes([0.0, 0.15, 0.3, 0.7], title="Original Image")
    ax_image = remove_axis_labels(ax_image)
    ax_mask_true = fig.add_axes([0.325, 0.15, 0.3, 0.7], title="True Mask")
    ax_mask_true = remove_axis_labels(ax_mask_true)
    ax_mask_pred = fig.add_axes([0.65, 0.05, 0.3, 0.9], title="Predicted Mask")
    ax_mask_pred = remove_axis_labels(ax_mask_pred)
    ax_image.imshow(image)
    im_true = ax_mask_true.imshow(mask_true, cmap=cmap, norm=norm)
    im_pred = ax_mask_pred.imshow(mask_predicted, cmap=cmap, norm=norm)
    colour_map = load_colour_map(Config.colour_map_path)
    formatter = plt.FuncFormatter(lambda val, loc: colour_map.get(val))
    fig.colorbar(
        im_pred, ticks=list(range(Config.output_channels)), format=formatter
    )
    plt.show()

def show_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset, num: int = 1
):
    """Prints image, segmentation mask and predictions.
    
    Args:
        model (tf.kera.Model) : Model used to make predictions.
        dataset (tf.data.Dataset) : Dataset for which prediction is made.
        num (int) : Number of elements to compare. 
    """
    for image, mask, weight in iter(dataset.take(num)):
        compare_model_predictions(model, image[0], mask[0])

########################################################################
# Load model
########################################################################

def load_model(path_to_model: str) -> tf.keras.Model:
    """Loads model from path.
    
    Args:
        path_to_model (str) : Path of where model is saved.
        
    Returns:
        model (tf.keras.Model) : Model loaded from path.
        """
    model = tf.keras.models.load_model(path_to_model)
    return model