import sys
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from Python import preprocessing


def load_image_dir_to_array(
    image_dir_path: str, sorted: bool = True, rgb: bool = True, normalised=False
) -> tuple[np.ndarray, list[str]]:
    """Loads all png images from dir into array with BGR encoding.
    
    Args:
        image_dir_path (str) : Path to directory containing .png images.
        sorted (bool) : If true, sorts images based on filenames.
        rgb (bool) : If true, loads images with BGR encoding, else trys to
            load image with 8-bit single-channel encoding.
        
    Returns:
        image_dataset (np.ndarray) : Array of images with dimensions: 
            image number, pixel row, pixel column, BGR channel.
        image_paths (list (str)) : List of image paths in the order which the
            images are loaded into array.
    """
    img_lookup = os.path.join(image_dir_path, '*.png')
    image_paths = glob.glob(img_lookup)
    if sorted:
        image_paths.sort()
    if not normalised:
        if rgb:
            images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in image_paths]
        else:
            images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths]
    else:
        images = [plt.imread(img, format="png") for img in image_paths]
    image_dataset = np.array(images, dtype=np.float32)
    if not rgb:
        image_dataset = np.expand_dims(image_dataset, axis=-1)
    return image_dataset, image_paths


def _pixel_rgb_to_index(pixel: np.ndarray, pixel_map: dict) -> np.ndarray:
    pixel_csv = ",".join(pixel.astype(str))
    out = np.array(pixel_map.get(pixel_csv), dtype=np.float32)
    return out


def dataset_rgb_to_index(
    image_dataset: np.ndarray,
    class_bgr_path: str,
    class_greyscale_path: str
) -> np.ndarray:
    """Transforms rgb encoded image datasets to index encoded image datasets.
    
    Args:
        image_datasets:(np.ndarray) : Array containing image dataset. Note:
            channels must be BGR encoded in the innermost axis.
        class_bgr_path (str) : Path to labelmap.txt file containing 
            label to rgb conversion.
        class_greyscale_path (str) : Path to labelmap.txt file containing
            label to index conversion."""
    rgb_to_index, index_to_rgb = preprocessing.get_rgb_index_maps(
        class_bgr_path, class_greyscale_path
    )
    map_rgb_to_index = lambda x : _pixel_rgb_to_index(x, rgb_to_index)
    image_dataset = np.apply_along_axis(
        map_rgb_to_index, axis=-1, arr=image_dataset
    )
    return image_dataset


def create_geryscale_masks(
    image_dir_path: str,
    class_bgr_path: str,
    class_greyscale_path: str
) -> None:
    """Create a directory with greyscale segmentation masks from a directory
    containing RGB segmentation masks.

    The greyscale directory will be located at the same location as the RGB 
    directory with '_greyscale' appended to the name.
    
    Args:
        image_dir_path (str) : Path to a directory containing .png 
            segmentation masks.
        class_bgr_path (str) : Path to labelmap.txt file mapping RGB 
            values to labels.
        class_greyscale_path (str) : Path to labelmap.txt file mapping index
            values to labels.
        
    Errors:
    FileExistsError: If a '_greyscale' directory already exists it will not be
        overwritten.
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


if __name__ == "__main__":
    # # Check input args
    # n_argv = len(sys.argv)
    # if n_argv != 4:
    #     use_msg = f"Usage: {sys.argv[0]} <path to BGR segmentation mask "
    #     use_msg += "directory> <path to BGR labelmap.txt file> <path to index labelmap.txt file>"
    #     print(use_msg)
    #     exit()
    # # Create greyscale segmentation masks
    # dir_path = sys.argv[1]
    # rgb_path = sys.argv[2]
    # index_path = sys.argv[3]
    dir_path = r"/home/vidarmarsh/CEZ_Mapping/Data/extra/SegmentationClass"
    bgr_path = r"/home/vidarmarsh/CEZ_Mapping/Data/extra/labelmap_bgr.txt"
    index_path = r"/home/vidarmarsh/CEZ_Mapping/Random/test_index_labelmap.txt"
    create_geryscale_masks(dir_path, bgr_path, index_path)

    