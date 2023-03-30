import os
import tensorflow as tf
class Config:
    """Database for configuring pipline
    
    Parameters:
        root_path (str) : Absolute path to project root dir.
    """
    root_path = os.path.split(os.path.split(__file__)[0])[0]
    data_path = os.path.join(root_path, "Data", "Priddy_processed")
    test_data_path = os.path.join(root_path, "Test_data")
    # tfdata_path = os.path.join(data_path, "TFDatasets")

    # Image Paths
    image_path = os.path.join(data_path, "JPEGImages_512")
    segmentation_path = os.path.join(
        data_path, "SegmentationClass_categorical_512"
    )
    # train_path = os.path.join(tfdata_path, "train_data")
    # val_path = os.path.join(tfdata_path, "val_data")
    # test_path = os.path.join(tfdata_path, "test_data")

    # Model Paths
    model_dir_path = os.path.join(data_path, "Models")
    weight_map_path = os.path.join(data_path, "weights.json")
    colour_map_path = os.path.join(data_path, "class_categorical.json")

    # Training Parameters
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
    output_channels = 11     # number of classes
    background_label = 0
    expansion_coeff = 2
    prefetch = tf.data.AUTOTUNE
