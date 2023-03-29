import numpy as np
from functools import reduce
from Python.load_numpy import load_image_dir_to_array
from Python.utils import write_dict_to_disk
from Python.config import Config


def get_weight_map(seg_array: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(seg_array, return_counts=True)
    unique, counts = [array.astype(float) for array in (unique, counts)]
    multiply = lambda x, y : x * y
    n_pixels = reduce(multiply, seg_array.shape)
    n_categories = len(unique)
    category_weights = n_pixels / (n_categories * counts)
    background_index = np.nonzero(unique == Config.background_label)
    category_weights[background_index] = 0
    weight_map = dict(zip(unique, category_weights))
    return weight_map

if __name__ == "__main__":
    weight_map_path = Config.weight_map_path
    mask_dir_path = Config.segmentation_path
    mask_array, mask_filenames = load_image_dir_to_array(mask_dir_path, sorted=True, rgb=False)
    weight_map = get_weight_map(mask_array)
    write_dict_to_disk(weight_map, weight_map_path)