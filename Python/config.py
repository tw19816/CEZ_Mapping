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

    # Image Paths
    image_path = os.path.join(data_path, "JPEGImages_512")
    segmentation_path = os.path.join(
        data_path, "SegmentationClass_categorical_512"
    )
    partition_path = os.path.join(data_path, "dataset_partition_2.json")

    # Model Paths
    model_dir_path = os.path.join(root_path, "Data", "Models")
    weight_map_path = os.path.join(data_path, "weights_2.json")
    colour_map_path = os.path.join(data_path, "class_categorical.json")

    # Training Parameters
    initial_learning_rate = 0.001
    decay_steps = 176
    decay_rate = 0.98
    decay_discrete = True
    batch_size = 4
    epochs = 250
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    shuffle_size = 1000
    input_shape = (512, 512, 4)
    channels_low = 32
    channels_high = 512
    middle_repeat = 8
    output_channels = 8     # number of classes
    background_label = 0    # class label to ignore during training
    prefetch = tf.data.AUTOTUNE
