import os
import numpy as np
from Python.config import Config
from Python.data_processing import image_tools

def _check_image_shape(image: np.ndarray, expected_shape: tuple) -> None:
    """Compare image shape to true shape and raise assertion error if 
        they are different.
    
    Args:
        image (np.ndarray) : Image arrays.
        expected_shape (tuple (int)) : Expected shape of image.
    
    Errors:
        AssertionError : If there is a mismatch between image shape and 
        expected shape.
    """
    errmsg = "dimension mismatch, expected {expected_shape} got {image.shape}"
    for dimension, true_dimension in zip(
        image.shape, expected_shape, strict=True
    ):
        assert dimension == true_dimension, errmsg


def test_load_image():
    print("Testing load_image")
    expected_rgba_shape = (1024, 1024, 4)
    expected_greyscale_shape = (1024, 1024, 1)
    # Check multi_channel image shapes
    rgba_path = os.path.join(
        Config.test_data_path,
        "data_processing",
        "sample_image_dir",
        "RGBA_image.png"
    )
    rgba_image = image_tools.load_image(rgba_path)
    _check_image_shape(rgba_image, expected_rgba_shape)
    # Check greyscale image shapes
    greyscale_path = os.path.join(
        Config.test_data_path,
        "data_processing",
        "sample_image_dir",
        "greyscale_image.png"
    )
    greyscale_image = image_tools.load_image(greyscale_path)
    _check_image_shape(greyscale_image, expected_greyscale_shape)
    # Check output is in range [0, 255] and not noramlised to [0, 1)
    assert rgba_image.max() <=255 and rgba_image.max() > 1, \
        "load_image output was normalised"
    # Check image output uses uint8 encoding
    assert np.issubdtype(rgba_image.dtype, np.uint8), \
        "load_image output is not uint8 encoded"


def test_save_image():
    print("Testing save_image")
    occupied_path = os.path.join(
        Config.test_data_path,
        "data_processing",
        "sample_rgba_image_dir",
        "post_process-32-20.png" 
    )
    available_path = os.path.join(
        Config.test_data_path,
        "data_processing",
        "sample_rgba_image_dir",
        "post_process-temp.png" 
    )
    bad_extension_path = os.path.join(
        Config.test_data_path,
        "data_processing",
        "sample_rgba_image_dir",
        "post_process-temp.jpeg" 
    )
    image = image_tools.load_image(occupied_path)
    # Test saving to occupied path raises a FileExistsError
    try:
        image_tools.save_image(image, occupied_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError(
            "does not throw FileExistsError when trying to write to occupied " +
            "path"
        )
    # Test saving to bad path which does not end in .png raises a 
    # ValueError
    try:
        image_tools.save_image(image, bad_extension_path)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Does not throw ValueError when trying to write to a path which " +
            "not end in .png"
        )
    # Test saving to a good path works by loading the image back and 
    # checking they are the same.
    image_tools.save_image(image, available_path)
    saved_image = image_tools.load_image(available_path)
    os.remove(available_path)
    assert (saved_image == image).all(), "saved image was corrupted"


def test_load_image_dir_to_array():
    print("Testing load_image_dir_to_array")
    expected_image_number = 2
    expected_image_shape = [1024, 1024, 4]
    image_dir_path = os.path.join(
        Config.test_data_path, "data_processing", "sample_rgba_image_dir"
    )
    image_array, image_paths = image_tools.load_image_dir_to_array(
        image_dir_path
    )
    # Check image_array has correct shape
    n_images = len(image_array)
    errmsg = f"found {n_images} but expected {expected_image_number} in "
    errmsg += f"{image_dir_path}"
    assert len(image_array) == expected_image_number, errmsg
    for image in image_array:
        _check_image_shape(image, expected_image_shape)
    # Check image_array pixels are is in range [0, 255] and not 
    # noramlised to [0, 1] when normalised is set to false
    assert image_array.max() <=255 and image_array.max() > 1, \
        "load image output was normalised, but should be un-normalised uint8"
    # Check image_array pixels are normalised to [0, 1] when normalised 
    # is set to True
    image_array, image_paths = image_tools.load_image_dir_to_array(
        image_dir_path, normalised=True
    )
    errmsg =  "load image output was not normalised to [0, 1] despite setting "
    errmsg += "normalised=True"
    assert image_array.max() <= 1, errmsg
    # Check image_paths are correct
    expected_image_paths = [
        os.path.join(image_dir_path, "post_process-32-20.png"),
        os.path.join(image_dir_path, "post_process-32-38.png")
    ]
    errmsg = "image_paths is different from expected image paths"
    for image_path, expected_path in zip(image_paths, expected_image_paths):
        assert image_path == expected_path, errmsg


if __name__ == "__main__":
    test_load_image()
    test_save_image()
    test_load_image_dir_to_array()