import os
import numpy as np
from PIL import Image
from glob import glob


def load_image(image_path: str) -> np.ndarray:
    """Load a single image into a numpy array with uint8 encoding.
    
    Args:
        image_path (str) : Path to image.
    
    Returns:
        image (np.ndarray (row, column, channel)) : Row major image 
            array. Note greyscale images are loaded into 3D numpy arrays
            with one element in the channel dimension.
    """
    with Image.open(image_path) as im:
        image = np.asarray(im)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Save an integer image array as a png image.
    
    Args:
        image (np.ndarray (int)) : Image array to save, must contain 
            integer pixel values.
        path (str) : Path to save location, must end in .png.
    
    Errors:
        FileExistsError : If path already exists.
        ValueError : If image path does not end in .png
        ValueError : If image dtype is not int
    """
    if os.path.exists(path):
        raise FileExistsError(f"file already exists at {path}")
    extension = path.split(".")[-1]
    if extension != "png":
        raise ValueError("path must end in .png")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError("image must be an interger array")
    Image.fromarray(image).save(path)


def load_image_dir_to_array(
    segmentation_rgb_dir_path: str, sorted: bool = True, normalised=False
) -> tuple[np.ndarray, list[str]]:
    """Loads all png images from dir into float32 array with RGB
        encoding.
    
    Args:
        image_dir_path (str) : Path to directory containing .png images.
        sorted (bool) : If true, sorts images based on filenames.
        
    Returns:
        image_dataset (np.ndarray (np.float32)) : Array of images with 
            dimensions: image number, pixel row, pixel column, RGB 
            channel.
        image_paths (list (str)) : List of image paths in the order 
            which the images are loaded into array.
    """
    img_lookup = os.path.join(image_dir_path, '*.png')
    image_paths = glob(img_lookup)
    if sorted:
        image_paths.sort()
    images = [load_image(img) for img in image_paths]
    image_dataset = np.array(images, dtype=np.float32)
    if normalised:
        image_dataset = image_dataset / 255.0
    return image_dataset, image_paths


def segmentation_masks_rgb_to_index(
    segmentation_masks: np.ndarray,
    class_to_rgb_path: str,
    class_to_greyscale_path: str
) -> np.ndarray:
    """Transforms rgb encoded image datasets to index encoded image datasets.
    
    Args:
        segmentation_maskss:(np.ndarray) : Array containing segmentation
            masks in RGB encoding with shape (image, row, column, 
            channel).
        class_to_rgb_path (str) : Path to labelmap.txt file containing 
            label to rgb conversion.
        class_to_greyscale_path (str) : Path to labelmap.txt file containing
            label to index conversion."""
    rgb_to_index, index_to_rgb = preprocessing.get_rgb_index_maps(
        class_to_rgb_path, class_to_greyscale_path
    )
    map_rgb_to_index = lambda x : _pixel_rgb_to_index(x, rgb_to_index)
    segmentation_masks = np.apply_along_axis(
        map_rgb_to_index, axis=-1, arr=segmentation_masks
    )
    return segmentation_masks


def create_greyscale_masks(
    segmentation_rgb_dir_path: str,
    output_dir_path: str,
    class_to_rgb_path: str,
    class_to_greyscale_path: str
) -> None:
    """Create a directory with greyscale segmentation masks from a 
    directory containing RGB segmentation masks.
    
    Args:
        segmentation_rgb_dir_path (str) : Path to a directory containing
            .png segmentation masks with RGB encoding.
        output_dir_path (str) : Path to output directory for greyscale
            images.
        class_to_rgb_path (str) : Path to labelmap.txt file mapping class
            labels to RGB values.
        class_to_greyscale_path (str) : Path to labelmap.txt file mapping 
            class labels to greyscale categorical values.
        
    Errors:
        FileExistsError : If output_dir_path already exists but is not a
            directory.
        FileExistsError: If an image already exists in the output 
            directory with the same name as an image in the input
            directory.
    """
    segmentation_array_rgb, image_paths = load_image_dir_to_array(
        segmentation_rgb_dir_path, sorted=True, normalised=False
    )
    segmentation_array_rgb = np.rint(segmentation_array_rgb).astype(np.uint8)
    segmentation_array_greyscale = segmentation_masks_rgb_to_index(
        segmentation_array_rgb,
        class_to_rgb_path,
        class_to_greyscale_path
    )
    if os.path.exists(output_dir_path):
        if not os.path.isdir(output_dir_path):
            raise FileExistsError(
                "output_dir_path already exists by it is not a directory"
            )
    else:
        os.mkdir(output_dir_path)
    for id, path in enumerate(image_paths):
        dir_path, filename = os.path.split(path)
        image_path_greyscale = os.path.join(output_dir_path, filename)
        save_image(segmentation_array_greyscale[id], image_path_greyscale)

