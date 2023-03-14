import os
class Config:
    """Database for configuring pipline
    
    Parameters:
        root_path (str) : Absolute path to project root dir.
    """
    root_path = os.path.split(os.path.split(__file__)[0])[0]
    image_path = os.path.join(
        root_path, "Data", "33", "JPEGImages"
    )
    segmentation_path = os.path.join(
        root_path, "Data", "33", "SegmentationClass_greyscale"
    )
    model_dir_path = os.path.joing(root_path, "Model")
    batch_size = 1
    epochs = 5
    train_size = 0.6
    val_size = 0.2
    test_size = 0.2
    input_shape = (1024, 1024, 4)
    channels_low = 48
    channels_high = 128
    middle_repeat = 0
    output_channels = 11