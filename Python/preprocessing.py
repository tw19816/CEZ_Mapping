import os
import numpy as np


def get_label_pixel_maps(path_to_labelmap: str) -> tuple[dict, dict]:
    """Create a map of labels to pixel values and inverse map from labelmap.txt
    file.
    
    Args:
        path_to_labelmap (str) : Path to labelmap.txt file.
    
    Returns:
        label_to_pixel (dict : str -> str) : Map of labels to csv pixel values.
        pixel_to_label (dict : str -> str) : Map of csv pixel values to labels.
    """
    with open(path_to_labelmap, "r") as f:
        lines = f.readlines()
    label_to_rgb = {}
    for id, line in enumerate(lines):
        if id == 0:
            continue
        label, rgb_value = line.split(":")[0:2]
        label_to_rgb[label] = rgb_value
    rgb_to_label = {}
    labels = label_to_rgb.keys()
    for label in labels:
        rgb_value = label_to_rgb[label]
        rgb_to_label[rgb_value] = label
    return label_to_rgb, rgb_to_label


def get_rgb_index_maps(
    path_to_rgb_labelmap: str, path_to_index_labelmap: str
) -> tuple:
    """Create a map from rgb values to geyscale index and inverse map from 
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
    label_to_rgb, rgb_to_label = get_label_pixel_maps(path_to_rgb_labelmap)
    label_to_index, index_to_label = get_label_pixel_maps(path_to_index_labelmap)
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