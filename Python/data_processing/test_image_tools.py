import os

from Python.config import Config
from Python.data_processing import image_tools

def _check_image_shape(image, true_shape):
    """Compare  to true shape and raise assertion error if 
        they are different.
    
    Args:
        image_path (str) : Path to image.
        true_shape (tuple (int)) : Expected shape of image
    
    Errors:
        AssertionError : If there is a mismatch between """
    errmsg = "dimension mismatch, expected {true_shape} got {image.shape}"
    for dimension, true_dimension in zip(
        image.shape, true_shape, strict=True
    ):
        assert dimension == true_dimension, errmsg

def test_load_image():
    rgba_path = os.path.join(
        Config.test_data_path, "data_processing", "RGBA_image.png"
    )
    rgba_image = image_tools.load_image(rgba_path)
    _check_image_shape(rgba_image, (1024, 1024, 4))
    greyscale_path = os.path.join(
        Config.test_data_path, "data_processing", "greyscale_image.png"
    )
    greyscale_image = image_tools.load_image(greyscale_path)
    _check_image_shape(greyscale_image, (1024, 1024, 1))


if __name__ == "__main__":
    test_load_image()