import os
import tensorflow as tf
class Config:
    """Database for configuring pipline
    
    Parameters:
        root_path (str) : Absolute path to project root dir.
    """
    root_path = os.path.split(os.path.split(__file__)[0])[0]
    data_path = os.path.join(root_path, "Data_combined")
    tfdata_path = os.path.join(data_path, "TFDatasets")
    image_path = os.path.join(data_path, "JPEGImages_256")
    segmentation_path = os.path.join(data_path, "SegmentationClass_greyscale_256")
    train_path = os.path.join(tfdata_path, "train_data")
    val_path = os.path.join(tfdata_path, "val_data")
    test_path = os.path.join(tfdata_path, "test_data")
    model_dir_path = os.path.join(data_path, "Models")
    weight_map_path = os.path.join(data_path, "weight_map1.json")
    colour_map_path = os.path.join(data_path, "colour_map.json")
    batch_size = 4
    epochs = 40
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    shuffle_size = 1000
    input_shape = (512, 512, 4)
    channels_low = 32
    channels_high = 512
    middle_repeat = 8
    output_channels = 11
    background_label = 0
    expansion_coeff = 2
    prefetch = tf.data.AUTOTUNE
