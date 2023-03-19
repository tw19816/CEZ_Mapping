import tensorflow as tf
import numpy as np
from Python.data_generator import generate_image_dataset_from_files
from Python.data_generator import get_dataset_from_generator
from Python.data_generator import Data_Generator
from Python.segmentation_model import deeplabv3plus
from Python.utils import get_png_paths_from_dir
from Python.utils import load_weight_map
from Python.utils import write_model_to_disk
from Python.config import Config


if __name__ == "__main__":
    image_dir = Config.image_path
    mask_dir = Config.segmentation_path
    image_files, mask_files = [
        get_png_paths_from_dir(dir) for dir in (image_dir, mask_dir)
    ]
    # need to add splitting of datasets
    weight_map = load_weight_map(Config.weight_map_path)
    # dataset = generate_image_dataset_from_files(
    #     image_files,
    #     mask_files,
    #     Config.batch_size,
    #     Config.shuffle_size, 
    #     weight_map,
    #     Config.output_channels
    # )
    dataset = Data_Generator(
        image_files, mask_files, 
        weight_map, 
        Config.input_shape, 
        Config.batch_size, 
        4, 
        Config.output_channels, 
        shuffle=True
    )

    model = deeplabv3plus(
        Config.input_shape, 
        Config.batch_size, 
        Config.output_channels,
        Config.channels_low,
        Config.channels_high,
        Config.middle_repeat
    )
    model.compile(
        optimizer='adam',
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ["accuracy", tf.keras.metrics.OneHotIoU(Config.output_channels, tuple(range(Config.output_channels)), sparse_y_pred=False)]
    )

    model_history = model.fit(
        dataset,
        epochs=Config.epochs,
        shuffle=True
    )
    write_model_to_disk(model, model_history, Config)



