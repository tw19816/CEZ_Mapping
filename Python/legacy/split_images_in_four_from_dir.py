import sys
from Python.utils import split_images_in_four_from_dir

if __name__ == "__main__":
    N_INPUTS = 3
    if (n_inputs := len(sys.argv)) != N_INPUTS:
        print(f"Usage: {sys.argv[0]} <Path to source dir> <path to output dir>")
        exit()
    source_path = sys.argv[1]
    dest_path = sys.argv[2]
    split_images_in_four_from_dir(source_path, dest_path)