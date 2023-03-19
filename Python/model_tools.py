import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Python.utils import load_image
from Python.config import Config
from Python.utils import load_weight_map
from glob import glob

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
        image = (image * 255).astype(np.int8)
    dataset = tf.data.Dataset.from_tensors(image)
    dataset = dataset.batch(1)
    mask = model.predict(dataset)
    normalise = lambda x: x/x.mean()
    # mask = np.apply_along_axis(normalise, arr=mask, axis=-1)
    mask = one_hot_to_categorical(mask)
    mask = np.squeeze(mask)
    return mask


def compare_model_predictions(model_path: str, image_path: str, mask_path: str):
    model = load_model(model_path)
    image = load_image(image_path)
    mask_true = load_image(mask_path)
    mask_predicted = model_predict_from_file(model, image_path)
    fig = plt.figure(figsize=[14, 6])
    fig.suptitle("Model performance")
    ax_image = fig.add_axes([0.05, 0.1, 0.25, 0.8], title="Original Image")
    ax_mask_true = fig.add_axes([0.35, 0.1, 0.25, 0.8], title="True Mask")
    ax_mask_pred = fig.add_axes([0.65, 0.1, 0.25, 0.8], title="Predicted Mask")
    ax_image.imshow(image)
    ax_mask_true.imshow(mask_true, vmin=0, vmax=10)
    ax_mask_pred.imshow(mask_predicted, vmin=0, vmax=10)
    plt.show()



def generate_weights(
    mask_array: np.ndarray, 
    weight_map: dict
) -> np.ndarray:
    weight_array = np.zeros(mask_array.shape, dtype=np.float32)
    for key in weight_map.keys():
        weight_array[mask_array == key] = weight_map.get(key)
    return weight_array

class Data_Generator(tf.keras.utils.Sequence):
    def __init__(self, 
        image_files: list[str],
        mask_files: list[str],
        weight_map: dict,
        output_size: tuple[int, int],
        batch_size: int,
        image_channels: int,
        n_classes: int,
        dtype: type = np.float32,
        shuffle: bool = False
    ):
        self.image_files = image_files
        self.mask_files = mask_files
        self.weight_map = weight_map
        self.output_size = output_size
        self.image_channels = image_channels
        self.n_classes = n_classes
        self.dtype = dtype
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
        # Error checking
        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Must have same number of image and mask files")
    
    def __len__(self):
        return int(len(self.image_files) / self.batch_size)

    def on_epoch_end(self):
        self.indices = np.arange(len(self) * self.batch_size)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise tf.errors.OutOfRangeError()
        image_array, mask_array, weight_array = [
            np.zeros(
                (self.batch_size, *self.output_size[0:2], channels), dtype=self.dtype
            ) for channels in (self.image_channels, self.n_classes, 1)
        ]
        indices = self.indices[
            index * self.batch_size: (index + 1) * self.batch_size
        ]
        for i, data_index in enumerate(indices):
            files = (self.image_files[data_index], self.mask_files[data_index])
            image, mask = map(load_image, (files))
            image = image / 255.0
            weights = generate_weights(mask, self.weight_map)
            mask = categorical_to_one_hot(mask, self.n_classes, self.dtype)
            image_array[i] = image
            mask_array[i] = mask
            weight_array[i] = weights
        return image_array, mask_array, weight_array


if __name__ == "__main__":
    path_to_model = r"/home/vidarmarsh/CEZ_Mapping/Data_combined/Models/3/model"
    path_to_image = r"/home/vidarmarsh/CEZ_Mapping/Data_combined/JPEGImages_256/post_process-34-40_10.png"
    path_to_mask = r"/home/vidarmarsh/CEZ_Mapping/Data_combined/SegmentationClass_greyscale_256/post_process-34-40_10.png"
    # compare_model_predictions(path_to_model, path_to_image, path_to_mask)
    images = glob(os.path.join(Config.image_path, "*.png"))
    masks = glob(os.path.join(Config.segmentation_path, "*.png"))
    dataset = Data_Generator(
        images, 
        masks, 
        load_weight_map(Config.weight_map_path),
        Config.input_shape, 
        1, 
        4, 
        Config.output_channels,
    )
    fig = plt.figure(figsize=[14, 6])
    fig.suptitle("Model performance")
    ax_image = fig.add_axes([0.05, 0.1, 0.25, 0.8], title="Original Image")
    ax_mask_true = fig.add_axes([0.35, 0.1, 0.25, 0.8], title="True Mask")
    ax_mask_pred = fig.add_axes([0.65, 0.1, 0.25, 0.8], title="Predicted Mask")
    image, mask_one_hot, weight = dataset[2]
    ax_image.imshow(np.squeeze(image))
    mask_categorical = np.squeeze(one_hot_to_categorical(mask_one_hot))
    mask_true = load_image(path_to_mask)
    ax_mask_true.imshow(mask_true, vmin=0, vmax=10)
    ax_mask_pred.imshow(mask_categorical, vmin=0, vmax=10)
    plt.show()
    