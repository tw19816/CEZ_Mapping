import numpy as np
import matplotlib.pyplot as plt

def display_mask(image_arr: np.ndarray, image_name: str) -> None:
    plt.imshow(image_arr)
    plt.title(image_name)
    plt.show()


def display_image_from_file(filepath):
    image = plt.imread(filepath)
    image = np.rint(image * 255)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    filepath = r"/home/vidarmarsh/CEZ_Mapping/Data_combined/SegmentationClass_greyscale/post_process-34-40.png"
    display_image_from_file(filepath)
