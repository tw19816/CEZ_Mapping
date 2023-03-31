import os
import numpy as np
from Python.data_processing.utils import segmentation_masks_rgb_to_index
from Python.data_processing.utils import split_dataset_paths
from Python.data_processing.utils import get_class_pixel_maps
from Python.data_processing.image_tools import load_image_dir_to_array
from Python.config import Config


def test_segmentation_masks_rgb_to_index():
    image_dir_path = os.path.join(
        Config.test_data_path, "data_processing", "sample_rgb_mask_dir"
    )
    path_to_rgb_map = os.path.join(
        Config.root_path, "Data", "Priddy_processed", "class_RGB.json"
    )
    path_to_index_map = os.path.join(
        Config.root_path, "Data", "Priddy_processed", "class_categorical.json"
    )
    image_dataset, image_filenames = load_image_dir_to_array(
        image_dir_path, sorted=False
    )
    image_dataset = image_dataset[0:3]
    image_dataset = segmentation_masks_rgb_to_index(
        image_dataset, path_to_rgb_map, path_to_index_map
    )
    assert image_dataset.shape == (3, 1024, 1024, 1), \
        "Array should have shape (3, 1024, 1024, 1) but has shape " + \
            f"{image_dataset.shape}"
    label_index, index_label = get_class_pixel_maps(
        path_to_index_map
    )
    correct_pixel_values = index_label.keys()
    for i in range(50):
        upper_bounds = image_dataset.shape
        random_id = tuple(
            [np.random.randint(0, upper) for upper in upper_bounds]
        )
        pixel_value = image_dataset[random_id]
        assert str(int(pixel_value)) in correct_pixel_values, \
            f"Pixel {random_id} had value {pixel_value} which is not in " + \
                "{correct_pixel_values}"


def test_split_dataset_paths():
    image_paths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mask_paths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_size = 0.6
    val_size = 0.2
    test_size = 0.2
    train, validation, test = split_dataset_paths(
        image_paths, mask_paths, train_size, val_size, test_size
    )
    assert len(train[0]) == 6
    assert len(validation[0]) == 2
    assert len(test[0]) == 2

if __name__ == "__main__":
    test_segmentation_masks_rgb_to_index()
    test_split_dataset_paths()