import os
import sys
from Python.config import Config

def conv_rgb_bgr_labelmap(path: str) -> None:
    """Create a BGR labelmap file from an rgb labelmap file.
    
    The BGR labelmap file will be stored in the same location as the RGB
    labelmap file with the same filename as the RGB labelmap + '_bgr'. 
    
    Args:
        path (str) : The path to a labelmap.txt file with RGB encoding
    
    Errors:
        FileExistsError : If the labelmap_bgr.txt file already exists then 
            it will not be overwritten and the function will fail.
    """
    dir_path, filename_rgb = os.path.split(path)
    with open(path, "r") as f:
        lines = f.readlines()
    for id, line in enumerate(lines):
        if id == 0:
            continue
        label, rgb_value = line.split(":")[0:2]
        rgb_reversed = rgb_value.split(",")[::-1]
        bgr_value = ",".join(rgb_reversed)
        line_bgr = ":".join([label, bgr_value, "", ""])
        lines[id] = line_bgr + '\n'
    head, extension = filename_rgb.split(".")
    filename_bgr = head + "_bgr" + "." + extension
    path_bgr = os.path.join(dir_path, filename_bgr)
    with open(path_bgr, "x") as f:
        f.writelines(lines)


if __name__ == "__main__":
    n_args = len(sys.argv)
    if n_args != 2:
        print(f"Usage: {sys.argv[0]} <path to rgb labelmap.txt>")
        exit()
    conv_rgb_bgr_labelmap(sys.argv[1])