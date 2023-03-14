import numpy as np
import matplotlib.pyplot as plt

def display_mask(image_arr: np.ndarray, image_name: str) -> None:
    plt.imshow(image_arr)
    plt.title(image_name)
    plt.show()