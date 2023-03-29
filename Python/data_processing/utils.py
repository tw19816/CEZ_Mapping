import os
from glob import glob
import numpy as np
import json

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


def load_weight_map(path: str) -> dict:
    """Loads class weights from json file.

    Args:
        path (str) : The path of the weight map.

    Returns:
        weight_map (dict) : A dictionary that maps classes to 
            class weights.
    """
    with open(path, "r") as file:
        weight_map = json.load(file)
    keys = np.array(tuple(weight_map.keys()), dtype=np.float32)
    values = np.array(
        [weight_map.get(key) for key in weight_map.keys()], dtype=np.float32
    )
    weight_map = dict(zip(keys, values))
    return weight_map