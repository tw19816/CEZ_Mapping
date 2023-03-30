import os
import numpy as np
from Python import preprocessing
from Python.load_numpy import load_image_dir_to_array, dataset_rgb_to_index
from Python.config import Config


def test_segmentation_masks_rgb_to_index():
    image_dir_path = os.path.join(
        Config.root_path, "Data", "33", "SegmentationClass"
    )
    path_to_rgb_map = os.path.join(
        Config.root_path, "Data", "33", "labelmap_bgr.txt"
    )
    path_to_index_map = os.path.join(
        Config.root_path, "Random", "test_index_labelmap.txt"
    )
    image_dataset, image_filenames = load_image_dir_to_array(
        image_dir_path, sorted=False
    )
    image_dataset = image_dataset[0:3]
    image_dataset = segmentation_masks_rgb_to_index(
        image_dataset, path_to_rgb_map, path_to_index_map
    )
    assert image_dataset.shape == (3, 1024, 1024, 1), \
        f"Array should have shape (3, 1024, 1024, 1) but has shape {image_dataset.shape}"
    label_index, index_label = preprocessing.get_label_pixel_maps(
        path_to_index_map
    )
    correct_pixel_values = index_label.keys()
    for i in range(50):
        upper_bounds = image_dataset.shape
        random_id = tuple([np.random.randint(0, upper) for upper in upper_bounds])
        pixel_value = image_dataset[random_id]
        assert str(int(pixel_value)) in correct_pixel_values, \
            f"Pixel {random_id} had value {pixel_value} which is not in {correct_pixel_values}"


if __name__ == "__main__":
    test_segmentation_masks_rgb_to_index()
    