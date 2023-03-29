import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
from datetime import datetime
from Python.config import Config


class Save_Class:
    def __init__(self, class_name, time_and_date, contents):
        self.class_name = class_name
        self.time_and_date = time_and_date
        self.contents = contents


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


def load_image(image_path: str) -> np.ndarray:
    """Load a single image into a numpy array with uint8 encoding.
    
    Args:
        image_path (str) : Path to image.
    
    Returns:
        image (np.ndarray (row, column, channel)) : Row major image array.
    """
    with Image.open(image_path) as im:
        image = np.asarray(im)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image


def split_image_in_four(image_path: str, save_dir: str) -> None:
    image = load_image(image_path)
    image = np.squeeze(image)
    width, height = image.shape[0:2]
    half_height, half_width = [int(length/2) for length in (width, height)]
    image00 = image[0:half_width, 0:half_height]
    image01 = image[0:half_width, half_height:height] 
    image10 = image[half_width:width, 0:half_height]
    image11 = image[half_width:width, half_height:height]
    images = [image00, image01, image10, image11]
    dir_path, image_file = os.path.split(image_path)
    image_name, image_type = image_file.split(".")
    for image_section, index in zip(images, ("00", "01", "10", "11")):
        filename = image_name + f"_{index}." + image_type
        filepath = os.path.join(save_dir, filename)
        if os.path.isfile(filepath):
            raise FileExistsError(f"Image file already exists: {filepath}")
        else:
            Image.fromarray(image_section).save(filepath)


def combine_images(save_path: str, source_dir: str) -> None:
    save_dir, filename = os.path.split(save_path)
    image_name, image_type = filename.split(".")
    images = []
    for index in ("00", "01", "10", "11"):
        image_section_path = os.path.join(source_dir, image_name)
        image_section_path += f"_{index}." + image_type
        image_section = load_image(image_section_path)
        images.append(image_section)
    image0 = np.concatenate((images[0], images[1]), axis=1)
    image1 = np.concatenate((images[2], images[3]), axis=1)
    image = np.concatenate((image0, image1), axis=0)
    image = np.squeeze(image)
    if os.path.isfile(save_path):
        raise FileExistsError(f"Image file already exists: {save_path}") 
    Image.fromarray(image).save(save_path)


def split_images_in_four_from_dir(image_dir: str, save_dir: str):
    image_search = os.path.join(image_dir, "*.png")
    image_paths = glob(image_search)
    for path in image_paths:
        split_image_in_four(path, save_dir)


def serialise_class_dict(item: object):
    keys = item.__dict__.keys()
    item_dict = {}
    for key in keys:
        if key[0:2] == "__":
            continue
        item_dict[key] = getattr(item, key)
    return item_dict


def write_dict_to_disk(dictionary: dict, path):
    save_json = json.dumps(dictionary, indent="\t")
    with open(path, "x") as file:
        file.write(save_json)


def write_class_to_disk(item: object, class_name: str, path: str):
    item_dict = serialise_class_dict(item)
    time_and_date = datetime.now().strftime("%d/%m/%Y %H:%M")
    save = Save_Class(class_name, time_and_date, item_dict)
    save_dict = serialise_class_dict(save)
    write_dict_to_disk(save_dict, path)


def write_model_to_disk(model: tf.keras.Model, history: tf.keras.callbacks.History, config: Config):
    index = 0
    while True:
        dir_path = os.path.join(config.model_dir_path, str(index))
        if not os.path.exists(dir_path):
            break
        index += 1
    os.mkdir(dir_path)

    model_save_path = os.path.join(dir_path, "model")
    model.save(model_save_path)

    config_save_path = os.path.join(dir_path, "config.json")
    write_class_to_disk(config, "config", config_save_path)
    
    history_save_path = os.path.join(dir_path, "history.json")
    write_dict_to_disk(history.history, history_save_path)


def write_dataset_to_disk(
    config: Config,
    train: tf.data.Dataset,
    val: tf.data.Dataset,
    test: tf.data.Dataset
):
    index = 0
    while True:
        dir_path = os.path.join(config.tfdata_path, str(index))
        if not os.path.exists(dir_path):
            break
        index += 1
    os.mkdir(dir_path)

    config_save_path = os.path.join(dir_path, "config.json")
    config_output = write_class_to_disk(config, "config", config_save_path)
    
    train_path = os.path.join(dir_path, "train")
    train.save(train_path)
    del train
    
    val_path = os.path.join(dir_path, "validation")
    val.save(val_path)
    del val
    
    test_path = os.path.join(dir_path, "test")
    test.save(test_path)
    del test


def load_weight_map(path):
    with open(path, "r") as file:
        weight_map = json.load(file)
    keys = np.array(tuple(weight_map.keys()), dtype=np.float32)
    values = np.array([weight_map.get(key) for key in weight_map.keys()], dtype=np.float32)
    weight_map = dict(zip(keys, values))
    return weight_map


def load_colour_map(path):
    with open(path, "r") as file:
        colour_map = json.load(file)
    keys = np.array(tuple(colour_map.keys()), dtype=float)
    values = [colour_map.get(key) for key in colour_map.keys()]
    colour_map = dict(zip(keys, values))
    return colour_map


def get_png_paths_from_dir(dir_path: str) -> list[str]:
    png_key = os.path.join(dir_path, "*.png")
    image_paths = glob(png_key)
    return image_paths