import os
import tensorflow as tf
from Python.segmentation_model import deeplabv3plus
from Python.data_pipeline import data_pipeline
from Python.load_numpy import load_image_dir_to_array
from Python.config import Config


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
        Config.batch_size, 
        Config.train_size, 
        Config.val_size, 
        Config.test_size,
        normalise_images=False
    )
    print(train_dataset.cardinality())
    del image_array, segmentation_array
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
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model_history = model.fit(
        train_dataset.prefetch(tf.data.AUTOTUNE),
        epochs=Config.epochs,
        validation_data=val_dataset,
    )
    model_name = "first_run"
    model_path = os.path.join(Config.model_dir_path, model_name)
    model.save(model_path, overwrite=False)