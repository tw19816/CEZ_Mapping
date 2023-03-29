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


def load_image_dir_to_array(
    image_dir_path: str, sorted: bool = True, normalised=False
) -> tuple[np.ndarray, list[str]]:
    """Loads all png images from dir into float32 array with RGB
        encoding.
    
    Args:
        image_dir_path (str) : Path to directory containing .png images.
        sorted (bool) : If true, sorts images based on filenames.
        
    Returns:
        image_dataset (np.ndarray) : Array of images with dimensions: 
            image number, pixel row, pixel column, RGB channel.
        image_paths (list (str)) : List of image paths in the order which the
            images are loaded into array.
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



def create_greyscale_masks(
    image_dir_path: str,
    output_dir_path: str,
    class_bgr_path: str,
    class_greyscale_path: str
) -> None:
    """Create a directory with greyscale segmentation masks from a 
    directory containing RGB segmentation masks.
    
    Args:
        image_dir_path (str) : Path to a directory containing .png 
            segmentation masks.
        output_dir_path (str) : Path to output directory for greyscale
            images.
        class_bgr_path (str) : Path to labelmap.txt file mapping RGB 
            values to labels.
        class_greyscale_path (str) : Path to labelmap.txt file mapping 
            index values to labels.
        
    Errors:
        FileExistsError: If an image already exists in the output 
            directory with the same name as an image in the input
            directory.
    """
    image_dataset_rgb, image_paths = load_image_dir_to_array(image_dir_path, sorted=False)
    image_dataset_rgb = np.rint(image_dataset_rgb).astype(np.uint8)
    image_dataset_greyscale = dataset_rgb_to_index(
        image_dataset_rgb,
        class_bgr_path,
        class_greyscale_path
    )
    image_dir = os.path.split(image_paths[0])[0]
    image_dir_greyscale = image_dir + "_greyscale"
    if not os.path.isdir(image_dir_greyscale):
        os.mkdir(image_dir_greyscale)
    else:
        raise FileExistsError(
            f"Cannot make greyscale directory since it already exists: {image_dir_greyscale}"
        )
    for id, path in enumerate(image_paths):
        dir_path, filename = os.path.split(path)
        image_path_greyscale = os.path.join(image_dir_greyscale, filename)
        cv2.imwrite(image_path_greyscale, image_dataset_greyscale[id])

