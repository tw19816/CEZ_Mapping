import os
import json
import numpy as np
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
        value = np.asarray(raw_class_pixel.get(key))
        if len(value) > 1:
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


def segmentation_masks_rgb_to_index(
    segmentation_masks: np.ndarray,
    class_to_rgb_path: str,
    class_to_greyscale_path: str
) -> np.ndarray:
    """Transforms rgb encoded segmentation masks to index encoded segmentation masks.
    
    Args:
        segmentation_maskss:(np.ndarray) : Array containing segmentation
            masks in RGB encoding with shape (image, row, column, 
            channel).
        class_to_rgb_path (str) : Path to labelmap.txt file containing 
            label to rgb conversion.
        class_to_greyscale_path (str) : Path to labelmap.txt file containing
            label to index conversion."""
    rgb_to_index, index_to_rgb = get_rgb_index_maps(
        class_to_rgb_path, class_to_greyscale_path
    )
    map_rgb_to_index = lambda x : _pixel_rgb_to_index(x, rgb_to_index)
    segmentation_masks = np.apply_along_axis(
        map_rgb_to_index, axis=-1, arr=segmentation_masks
    )
    return segmentation_masks
