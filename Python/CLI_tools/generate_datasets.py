import os
import tensorflow as tf
from Python.data_pipeline import data_pipeline
from Python.load_numpy import load_image_dir_to_array
from Python.config import Config
from Python.utils import write_dataset_to_disk

if __name__ == "__main__":
    image_array, image_paths = load_image_dir_to_array(
        Config.image_path, sorted=True, rgb=True, normalised=True
    )
    segmentation_array, segmentation_paths= load_image_dir_to_array(
        Config.segmentation_path, sorted=True, rgb=False
    )
    train_dataset, val_dataset, test_dataset = data_pipeline(
        image_array, 
        segmentation_array, 
        Config.train_size, 
        Config.val_size,
        Config.test_size,
        normalise_images=False
    )
    del image_array, segmentation_array
    print("Writing data to disk")
    write_dataset_to_disk(Config, train_dataset, val_dataset, test_dataset)