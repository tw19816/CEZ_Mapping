import tensorflow as tf
import numpy as np
import sys


def load_image_and_mask(
    image_path: str, mask_path: str
) -> tuple[tf.Tensor, tf.Tensor]:
    """Loads image and segmentation masks from png to tf.Tensors.
    
    Args:
        image_path (str) : Path of an image.
        mask_path (str) : Path of a segmentation mask.
        
    Returns: 
        image (tf.Tensor (tf.float32)) : Image tensor with dimensions 
            [row, column, channel],
        mask (tf.Tensor (uint8)) : Segmentation mask tensor dimensions
            [row, column, channel]      
    """
    image_file, mask_file = [
        tf.io.read_file(file) for file in (image_path, mask_path)
    ]
    image = tf.image.decode_png(image_file, channels=4, dtype=tf.uint8)
    mask = tf.image.decode_png(mask_file, channels=1, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, mask


def _make_weight_mask(
    image_array: tf.Tensor,
    mask_array: tf.Tensor, 
    class_weights: np.ndarray
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Makes weighted mask from segmentation mask and class weights.
    
    Args:
        image_array (tf.Tensor) : Image tensor.
        mask_array (tf.Tensor) : Segmentation mask tensor.
        class_weights (np.ndarray) : Class weighting.
    
    Returns:
        image_array (tf.Tensor) : Image tensor.
        mask_array (tf.Tensor) : Segmentation mask tensor.
        weight_mask (tf.Tensor) : Weight mask.
        """
    class_weights_tensor = tf.constant(class_weights)
    weight_mask = tf.gather(
        class_weights_tensor, indices=tf.cast(mask_array, tf.int32)
    )
    return image_array, mask_array, weight_mask


def generate_image_dataset_from_files(
    image_files: list[str],
    mask_files: list[str],
    batch_size: int,
    prefetch: int,
    shuffle_size: int,
    weights: np.ndarray
) -> tf.data.Dataset:
    """Generates a dataset which returns batches of images, segmentation
        masks and weight masks. 

    Args:
        image_files (list (str)) : List of image paths.
        mask_files (list (str)) : List of segmentation mask paths.
        batch_size (int) : Batch size.
        prefetch (int) : Number of batches to prefetch.
        shuffle_size (int) : Size of shuffle buffer.
        weights (np.ndarray) : Array of class weights.

    Returns:
        dataset (tf.data.Dataset) : Dataset of images, segmentation
            masks and weight masks.

    """
    n_images = len(image_files)
    # Check number of images and segmentation masks is the same
    if n_images != (n_masks := len(mask_files)):
        errmsg = f"different number of image and mask files, found {n_images}"
        errmsg += f"image files and {n_masks} mask files" 
        raise ValueError(errmsg)
    
    weights = weights.astype(np.float32)

    def make_weight_mask(
        image_array: tf.Tensor,
        mask_array: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Maps images, segmentation masks to images, segmentation masks
            and weight mask.
        
        Args:
        image_array (tf.Tensor) : Image tensor.
        mask_array (tf.Tensor) : Segmentation mask tensor.

        Returns:
            image_array (tf.Tensor) : Image tensor.
            mask_array (tf.Tensor) : Segmentation mask tensor.
            weight_mask (tf.Tensor) : Weight mask.
        """
        return _make_weight_mask(image_array, mask_array, weights)
    
    # Creates dataset of filenames
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.shuffle(shuffle_size)
    # Maps filenames -> images, masks
    dataset = dataset.map(
        load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    # Maps images, mask -> images, mask, weights
    dataset = dataset.map(
        make_weight_mask, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(prefetch)
    return dataset


def _augment_datapoint(
    image: tf.Tensor, mask: tf.Tensor, rng: np.random.Generator
) -> tuple[tf.Tensor, tf.Tensor]:
    """Creates random augmentations to the data. 
    
    The possible augmentations are flip image vertically and flip image 
    horizontally.
    
    Args:
        image (tf.Tensor) : Image tensor.
        mask (tf.Tensor) : Segmentation mask tensor.
        rng (np.random.Generator) : Numpy random number generator.

    Returns:
        image (tf.Tensor) : Augmented image tensor.
        mask (tf.Tensor) : Augmented segmentation mask tensor.
    """
    augment = rng.uniform(size=2) >= 0.5
    if augment[0]:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if augment[1]:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    return image, mask


def augment_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Augments the dataset with transformations defined in 
        _augment_datapoint().
        
    Args:
        dataset (tf.data.Dataset) : Current dataset.

    Returns:
        dataset (tf.data.Dataset) : Augmented dataset.
    """
    rng = np.random.default_rng()
    def augment_datapoint(image, mask):
        return _augment_datapoint(image, mask, rng)
    dataset = dataset.map(
        augment_datapoint, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


def split_dataset(
    dataset: tf.data.Dataset,
    train_size: float,
    val_size: float,
    test_size: float
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Partitions a dataset into train validation and test sets.
    
    Args:
        dataset (tf.data.Dataset) : The dataset to split. This must 
            return (image, mask) when iterated across.
        train_size (float) : Fraction of dataset to use for training.
        val_size (float) : Fraction of dataset to use for validation.
        test_size (float) : Fraction of dataset to use for testing.
    
    Returns:
        train_dataset (tf.data.Dataset) : The dataset to use for 
            training.
        val_dataset (tf.data.Dataset) : The dataset to use for 
            validation.
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
