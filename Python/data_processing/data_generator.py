import tensorflow as tf
import numpy as np


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
    prefetch: int,
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


def _augment_datapoint(
    image: tf.Tensor, mask: tf.Tensor, rng: np.random.Generator
) -> tuple[tf.Tensor, tf.Tensor]:
    augment = rng.uniform(size=2) >= 0.5
    if augment[0]:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if augment[1]:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    return image, mask


def augment_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    rng = np.random.default_rng()
    def augment_datapoint(image, mask):
        return _augment_datapoint(image, mask, rng)
    dataset = dataset.map(augment_datapoint, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def split_dataset(
    dataset: tf.data.Dataset,
    train_size: float,
    val_size: float,
    test_size: float
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Partitions a dataset into train validation and test sets.
    
    Args:
        dataset (tf.data.Dataset) : The dataset to split. This must return
            (image, mask) when iterated across.
        train_size (float) : Fraction of dataset to use for training.
        val_size (float) : Fraction of dataset to use for validation.
        test_size (float) : Fraction of dataset to use for testing.
    
    Returns:
        train_dataset (tf.data.Dataset) : The dataset to use for training.
        val_dataset (tf.data.Dataset) : The dataset to use for validation.
        test_dataset (tf.data.Dataset) : The dataset to use for testing.
    """
    dataset_size = len(dataset)
    sum_partitions = train_size + val_size, test_size
    if np.allclose(sum_partitions, 1, atol=3*sys.float_info.epsilon):
        errmsg = f"Paritions should sum to 1 but sum to {sum_partitions}"
        raise ValueError(errmsg)
    
    ignore_weights = lambda x, y, z : [x, y]

    train_size = int(dataset_size * train_size)
    val_size = int(dataset_size * val_size)
    test_size = int(dataset_size * test_size)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    test_dataset = val_dataset.skip(val_size).map(
        ignore_weights, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.take(val_size).map(
        ignore_weights, num_parallel_calls=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
