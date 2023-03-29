import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Python.utils import load_image
from Python.utils import load_colour_map
from Python.config import Config
from Python.utils import load_weight_map

def load_model(path_to_model: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(path_to_model)
    return model


def one_hot_to_categorical(one_hot: tf.Tensor):
    get_categorical = lambda x : np.argmax(x)
    one_hot = np.apply_along_axis(get_categorical, axis=-1, arr=one_hot)
    return one_hot


def categorical_to_one_hot(
    categorical: np.ndarray, n_classes: int, dtype: type = None
) -> np.ndarray:
    if dtype == None:
        dtype = categorical.dtype
    map_one_hot = dict(
        zip(
        range(n_classes),
        np.zeros([n_classes, n_classes],
        dtype=dtype)
        )
    )
    for i in range(n_classes):
        map_one_hot[i][i] = 1
    one_hot_shape = list(categorical.shape)
    one_hot_shape[-1] = n_classes
    one_hot = np.zeros(one_hot_shape, dtype=dtype)
    for key in map_one_hot.keys():
        one_hot[np.squeeze(categorical == key)] = map_one_hot.get(key)
    return one_hot  


def model_predict_from_file(
    model: tf.keras.Model, filepath: str, normalised: bool = False
):
    image = plt.imread(filepath)
    if not normalised:
        image = (image * 255).astype(np.float32)
    dataset = tf.data.Dataset.from_tensors(image)
    dataset = dataset.batch(1)
    mask = model.predict(dataset)
    mask = one_hot_to_categorical(mask)
    mask = np.squeeze(mask)
    return mask



def remove_axis_labels(axis: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    axis.get_yaxis().set_visible(False)
    axis.get_xaxis().set_visible(False)
    return axis


def compare_model_predictions(model: tf.keras.Model, image, mask_true):
    dataset = tf.data.Dataset.from_tensors(image).batch(1)
    mask_predicted = model.predict(dataset)
    mask_predicted = np.squeeze(mask_predicted)
    mask_predicted = one_hot_to_categorical(mask_predicted)
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 11.5, 1), cmap.N)
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
    fig.colorbar(im_pred, ticks=list(range(Config.output_channels)), format=formatter)
    plt.show()


if __name__ == "__main__":
    path_to_model = r"/home/vidarmarsh/CEZ_Mapping/Data_combined/Models/4/model"
    path_to_image = r"/home/vidarmarsh/CEZ_Mapping/Data_combined/JPEGImages_256/post_process-34-40_11.png"
    path_to_mask = r"/home/vidarmarsh/CEZ_Mapping/Data_combined/SegmentationClass_greyscale_256/post_process-34-40_11.png"
    model = load_model(path_to_model)
    image = load_image(path_to_image)
    mask_true = load_image(path_to_mask)
    compare_model_predictions(path_to_model, path_to_image, path_to_mask)
