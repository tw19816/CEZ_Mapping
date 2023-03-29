import os
from glob import glob


def get_png_paths_from_dir(dir_path: str) -> list[str]:
    """Get path to all images.
    
    Args:
        dir_path (str) : The path to image directory.
        
    Returns:
        image_paths (list (str)) : List of all image paths.
    """
    png_key = os.path.join(dir_path, "*.png")
    image_paths = glob(png_key)
    return image_paths