import os
import tensorflow as tf
from Python.config import Config
from Python import preprocessing


def dataset_setup(directory: str, batch: int, val_split: float):
    '''Creates the necessary datasets with training and validation split val_split.
    Args:
        directory (str) : Directory of dataset.
        batch_size (int) : Number of inputs per batch.
        val_split (float) :  Float between 0 and 1, fraction of data to reserve for validation.

    Returns:
        train_dataset (tf.data.Dataset) : Tf.data.Dataset object containing training dataset.
        val_dataset (tf.data.Dataset) : Tf.data.Dataset object containing validation dataset.
    '''

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None,
        color_mode='rgb',
        batch_size=batch,
        image_size=(1024, 1024),
        shuffle=False,
        validation_split=val_split,
        subset='training',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None,
        color_mode='rgb',
        batch_size=batch,
        image_size=(1024, 1024),
        shuffle=False,
        validation_split=val_split,
        subset='validation',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        )
    return train_dataset, val_dataset


def rgb_to_index_element(
    dataset_element: tf.Tensor,
    rgb_to_index_map: dict,
) -> tf.data.Dataset:
    len_x, len_y, len_pixel = dataset_element.shape
    for idx in range(len_x):
        for idy in range(len_y):
            print(dataset_element[idx, idy])
            rgb_vals = ",".join(dataset_element[idx, idy])
            dataset_element[idx, idy] = rgb_to_index_map.get(rgb_vals)
    return dataset_element


def rgb_to_index(
    dataset: tf.data.Dataset,
    path_to_rgb_labelmap: str,
    path_to_index_labelmap: str
) -> tf.data.Dataset:
    """Converts rgb pixel values to index values for a segmentation dataset.
    
    Args:
        dataset (tf.data.Dataset) : A dataset of unbatched segmentation masks 
            with rgb pixel values.
        path_to_rgb_labelmap (str) : The path to a labelmap.txt file containing
            rgb to label conversions.
        parth_to_index_labemap (str) : The path to a labelmap.txt file
            containing index to label conversions.
    
    Returns:
        out (tf.data.Dataset) : A dataset of unbatched segmentation masks with
            index pixel values.
    """
    rgb_index, index_rgb = preprocessing.get_rgb_index_maps(
        path_to_rgb_labelmap, path_to_index_labelmap
    )
    for id, element in dataset.as_numpy_iterator():
        
    out = dataset.map(lambda x: rgb_to_index_element(x, rgb_index))
    return out


def normalise(dataset, pixels: int = 255):
    '''Normalises pixel values to float range 0 and 1.
    Args:
        dataset (tf.data.Dataset): TensorFlow dataset object with images encoded.
        pixels (int): Max pixel value.
    Returns:
        normalised_ds (tf.data.Dataset): TensorFlow dataset object with rescaled pixel value images encoded.
    '''

    normalisation_layer = tf.keras.layers.Rescaling(1./pixels)
    normalised_ds = dataset.map(lambda x: (normalisation_layer(x)))
    return normalised_ds

def data_pipeline(
        dir_path: str,
        batch: int,
        val_split: float,
        path_to_rgb_labelmap: str,
        path_to_index_labelmap: str,
        pixels: int = 255):
    '''Loads the image data and segmentation masks into a tf.data.Dataset 
    object.

    It splits the data into the training set and validation set with a 
    validation split (val_split). Then normalises the image data.
    
    Args:
        dir_path (str) : Dataset directory with two subdirectories: inputs
            containing images and targets containing pgn segmentation
            masks. Directory must also contain a labelmap.txt file with label
            and rgb pairs.
        batch_size (int) : Number of inputs per batch.
        val_split (float) :  Float between 0 and 1, fraction of data to reserve
            for validation.
        path_to_rgb_labelmap (str) : The path to a labelmap.txt file containing
            rgb to label conversions.
        parth_to_index_labemap (str) : The path to a labelmap.txt file
            containing index to label conversions.
        pixels (int): Max pixel value.

    Returns:
        norm_train_dataset (tf.data.Dataset) : Tf.data.Dataset object 
            containing training dataset.
        norm_val_dataset (tf.data.Dataset) : Tf.data.Dataset object 
            containing validation dataset.
        mask
    '''
    train = {}
    val = {}
    for sub_dir in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir)
        train_temp, val_temp = dataset_setup(sub_dir_path, batch, val_split)
        train[sub_dir] = train_temp
        val[sub_dir] = val_temp
    train["targets"] = rgb_to_index(
        train.get("targets"),
        path_to_rgb_labelmap,
        path_to_index_labelmap
    )
    val["targets"] = rgb_to_index(
        val.get("targets"),
        path_to_rgb_labelmap,
        path_to_index_labelmap
    )
    train = tf.data.Dataset.zip(train)
    val = tf.data.Dataset.zip(val)
    # train = train.get("inputs").concatenate(train.get("targets"))
    # val = val.get("inputs").concatenate(val.get("targets"))
    return train, val


if __name__ == "__main__":
    data_path = os.path.join(Config.root_path, "Data", "1_data")
    rgb_path = os.path.join(Config.root_path, "Random", "test_labelmap.txt")
    index_path = os.path.join(
        Config.root_path, "Random", "test_index_labelmap.txt"
    )
    train, val = data_pipeline(data_path, None, 0.2, rgb_path, index_path)
    for id, element in enumerate(train):
        print(id)