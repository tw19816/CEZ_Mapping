import os
import sys
import json
import random
import numpy as np
from functools import reduce
from glob import glob


def get_png_paths_from_dir(dir_path: str) -> list[str]:
    """Get path to all images.
    
    Args:
        dir_path (str) : The path to image directory.
        
    Returns:
        image_paths (list (str)) : List of all image paths.
    """
    png_key = os.path.join(dir_path, "*.png")
    image_paths = glob(png_key)
    return image_paths


def load_json_dict(path: str) -> dict:
    """Creates a python dictionary for json key-value pairs.
    
    Args:
        path (str) : Path to the json file.
        
    Returns:
        output (dict) : Dictionary containing json key-value pairs.
    """
    with open(path, "r") as file:
        output = json.load(file)
    return output
    

def load_weight_map(path: str) -> dict:
    """Loads class weights from json file to dict with np.float32 
        encoding.

    Args:
        path (str) : The path of the weight map.

    Returns:
        weight_map (dict) : A dictionary that maps classes to 
            class weights.
    """
    weight_map = load_json_dict(path)
    keys = np.array(tuple(weight_map.keys()), dtype=np.float32)
    values = np.array(
        [weight_map.get(key) for key in weight_map.keys()], dtype=np.float32
    )
    weight_map = dict(zip(keys, values))
    return weight_map


def get_class_pixel_maps(path: str) -> tuple[dict, dict]:
    """Create a map of class labels to pixel values and inverse map from
        json file. If the pixel values are saved as an array then these 
        will be coverted to csv strings.
    
    Args:
        path (str) : Path to json file relating classes (str) to 
            pixel values (int | float | array (int | float))
    
    Returns:
        class_to_pixel (dict : str -> str) : Map of class labels to 
            pixel values.
        pixel_to_class (dict : str -> str) : Map of pixel values to 
            class labels.
    """
    raw_class_pixel = load_json_dict(path)
    class_to_pixel = {}
    for key in raw_class_pixel.keys():
        value = np.asarray(raw_class_pixel.get(key), dtype=str)
        if value.shape != ():
            value = ",".join(value)
        else:
            value = str(value)
        class_to_pixel[key] = value
    pixel_to_class = {}
    for key in class_to_pixel.keys():
        value = class_to_pixel.get(key)
        pixel_to_class[value] = key
    return class_to_pixel, pixel_to_class


def get_rgb_index_maps(
    path_to_rgb_labelmap: str, path_to_index_labelmap: str
) -> tuple:
    """Create a map from rgb values to greyscale index and inverse map from 
    labelmap.txt files.
    
    Args:
        path_to_rgb_labelmap (str) : Path to labelmap.txt file containing 
            label to rgb conversion.
        path_to_index_labelmap (str) : Path to labelmap.txt file containing
            label to index conversion.
    
    Returns:
        rgb_to_index (dict: str -> int) : Map from csv rgb values to index 
            value.
        index_to_rgb (dict : int -> tuple (int)) : Map from index value to rgb
            values.
    """
    label_to_rgb, rgb_to_label = get_class_pixel_maps(path_to_rgb_labelmap)
    label_to_index, index_to_label = get_class_pixel_maps(path_to_index_labelmap)
    rgb_to_index = {}
    for rgb_value in rgb_to_label.keys():
        label = rgb_to_label.get(rgb_value)
        rgb_to_index[rgb_value] = int(label_to_index.get(label))
    index_to_rgb = {}
    indices = index_to_label.keys()
    for index in indices:
        label = index_to_label.get(index)
        rgb = np.array(label_to_rgb.get(label).split(","), dtype=int)
        index_to_rgb[int(index)] = rgb
    return rgb_to_index, index_to_rgb


def _pixel_rgb_to_index(pixel: np.ndarray, pixel_map: dict) -> np.ndarray:
    """Applies a pixel mapping to the channels of a single pixel in a 
        segmentation mask.
        
    Args:
        pixel (np.ndarray) : Array of pixel channels.
        pixel_map (dict) : mapping of pixel channels"""
    pixel_csv = ",".join(pixel.astype(str))
    out = np.array(pixel_map.get(pixel_csv), dtype=np.float32)
    return out


def segmentation_masks_rgb_to_index(
    segmentation_masks: np.ndarray,
    class_to_rgb_path: str,
    class_to_greyscale_path: str
) -> np.ndarray:
    """Transforms rgb encoded segmentation masks to index encoded segmentation masks.
    
    Args:
        segmentation_masks:(np.ndarray) : Array containing segmentation
            masks in RGB encoding with shape (image, row, column, 
            channel).
        class_to_rgb_path (str) : Path to json file containing label to 
            rgb conversion.
        class_to_greyscale_path (str) : Path to json file containing 
            label to categorical conversion.
        
    Returns:
        segmentation_mask_categorical (np.ndarray) : Categorical 
            segmentation masks.
    """
    rgb_to_index, index_to_rgb = get_rgb_index_maps(
        class_to_rgb_path, class_to_greyscale_path
    )
    segmentation_masks_categorical = np.full(
        segmentation_masks.shape[0:-1],
        np.nan,
        dtype=np.uint8)
    for index in index_to_rgb.keys():
        rgb = index_to_rgb.get(index)
        segmentation_masks_categorical[
            np.nonzero((segmentation_masks == rgb).all(axis=-1))
        ] = index
    segmentation_masks_categorical = np.expand_dims(
        segmentation_masks_categorical, axis=-1
    )
    return segmentation_masks_categorical


def create_weight_map(
    segmentation_array: np.ndarray
) -> dict:
    """Create a dictionary that maps greyscale categorical class labels
    to their respective weights.
    
    Args:
        segmentation_array (np.ndarray) : array containing all segmentation
            masks in the dataset.

    Returns:
        weight_map (dict) : Dictionary mapping float32 greyscale 
            categorical class values to float32 class weights.
    """
    unique, counts = np.unique(segmentation_array, return_counts=True)
    unique, counts = [array.astype(float) for array in (unique, counts)]
    multiply = lambda x, y : x * y
    n_pixels = reduce(multiply, segmentation_array.shape)
    n_categories = len(unique)
    category_weights = n_pixels / (n_categories * counts)
    weight_map = dict(zip(unique, category_weights))
    return weight_map


def split_dataset_paths(
    image_paths: list[str],
    mask_paths: list[str],
    train_size: float,
    validation_size: float,
    test_size: float
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Split image and segmentation mask filepaths into train, 
        validation, and test partitions.
    
    Args:
        image_paths (list (str)) : Filepaths to all images in the 
            dataset, note these must all be contained in the same 
            directory.
        mask_paths (list (str)) : Filepaths to all masks in the dataset,
            note these must all be contained in the same directory and
            must have the same filename
        train_size (float) : Fraction of data to use in the training 
            set.    
        validation_size (float) : Fraction of data to use in the
            validation set.    
        test_size (float) : Fraction of data to use in the test set.    
    
    Returns:
        train (list (tuple (str, str))) : List of image-mask pairs in 
            the training set.
        validation (list (tuple (str, str))) : List of image-mask pairs
            in the validation set.
        test (list (tuple (str, str))) : List of image-mask pairs in the
            test set. 
    
    Errors:
        ValueError : If train_size, validation_size, and test_size do
            not add up to one.
        ValueError : If image_paths and mask_paths contain a different
            number of elements
    """
    sum_partitions = train_size + validation_size, test_size
    if np.allclose(sum_partitions, 1, atol=3*sys.float_info.epsilon):
        errmsg = f"Paritions should sum to 1 but sum to {sum_partitions}"
        raise ValueError(errmsg)
    dataset = list(zip(image_paths, mask_paths, strict=True))
    length = len(dataset)
    random.shuffle(dataset)
    n_train, n_validation, n_test = [
        int(length*fraction) for fraction in [
            train_size, validation_size, test_size
        ]
    ]
    train = dataset[0:n_train]
    validation = dataset[n_train:n_train + n_validation]
    test = dataset[n_train + n_validation:n_train + n_validation + n_test]
    train, validation, test = [
        list(map(list, zip(*data))) for data in [train, validation, test]
    ]
    return train, validation, test



