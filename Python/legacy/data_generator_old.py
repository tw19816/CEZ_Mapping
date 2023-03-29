import tensorflow as tf
import numpy as np
from collections.abc import Callable
from Python.config import Config
from Python.model_tools import categorical_to_one_hot
from Python.utils import load_image


def load_image_and_mask(image_path, mask_path):
    image_file, mask_file = [
        tf.io.read_file(file) for file in (image_path, mask_path)
    ]
    image = tf.image.decode_png(image_file, channels=4, dtype=tf.uint8)
    mask = tf.image.decode_png(mask_file, channels=1, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, mask


def _make_weight_map(
    image_array: tf.Tensor,
    mask_array: tf.Tensor, 
    class_weights: np.array
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    class_weights_tensor = tf.constant(class_weights)
    weights = tf.gather(class_weights_tensor, indices=tf.cast(mask_array, tf.int32))
    return image_array, mask_array, weights


def generate_image_dataset_from_files(
    image_files: str,
    mask_files: str,
    batch_size: int,
    prefetch: int | tf.data.AUTOTUNE,
    shuffle_size: int,
    weights: np.ndarray
) -> tf.data.Dataset:
    n_images = len(image_files)
    if n_images != (n_masks := len(mask_files)):
        errmsg = f"different number of image and mask files, found {n_images}"
        errmsg += f"image files and {n_masks} mask files" 
        raise ValueError(errmsg)
    
    weights = weights.astype(np.float32)
    def make_weight_map(
        image_array: tf.Tensor,
        mask_array: tf.Tensor
    ):
        return _make_weight_map(image_array, mask_array, weights)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.map(
        load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        make_weight_map, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(prefetch)
    return dataset



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


def get_dataset_from_generator(
    image_files: list[str],
    mask_files: list[str],
    weight_map: dict,
    output_size: tuple[int, int],
    batch_size: int,
    image_channels: int,
    n_classes: int,
    dtype: type = np.float32,
    shuffle: bool = False
) -> tf.data.Dataset:

    dataset = tf.data.Dataset.from_generator(
        Data_Generator, 
        output_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ], 
        args=[
            image_files,
            mask_files,
            weight_map,
            output_size,
            batch_size,
            image_channels,
            n_classes,
            dtype,
            shuffle
        ])
    dataset.prefetch(1)
    return dataset
    



    


if __name__ == "__main__":
    elements = np.linspace(0, 100, 100)
    generator = Data_Generator(elements, elements, (10, 10))
    item = generator[1]