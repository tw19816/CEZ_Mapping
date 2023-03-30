import os
import json
from datetime import datetime
import tensorflow as tf

class Save_Class:
    def __init__(self, class_name, time_and_date, contents):
        self.class_name = class_name
        self.time_and_date = time_and_date
        self.contents = contents


def serialise_class_dict(item: object):
    """Serialises class object to json format.

    Args:
        item (object) : Object to be serialised.
    """
    keys = item.__dict__.keys()
    item_dict = {}
    for key in keys:
        if key[0:2] == "__":
            continue
        item_dict[key] = getattr(item, key)
    return item_dict


def write_dict_to_disk(dictionary: dict, path: str):
    """Writes dictionary to disk as json file.

    Args:
        dictionary (dict) : Dictionary to be saved.
        path (str) : Path of save location.
    """
    save_json = json.dumps(dictionary, indent="\t")
    with open(path, "x") as file:
        file.write(save_json)


def write_class_to_disk(item: object, class_name: str, path: str):
    """Writes class object to disk with day and time stamp.

    Args:
        item (object) : Class that will be saved to disk.
        class_name (str) : Name of class.
        path (str) : Path to save location.
    
    """
    item_dict = serialise_class_dict(item)
    time_and_date = datetime.now().strftime("%d/%m/%Y %H:%M")
    save = Save_Class(class_name, time_and_date, item_dict)
    save_dict = serialise_class_dict(save)
    write_dict_to_disk(save_dict, path)


def write_model_to_disk(
        model: tf.keras.Model, 
        history: tf.keras.callbacks.History, 
        parent_dir_path: str, 
        config: object
    ):
    """Saves a given model to disk with the configuration and history of 
        training parameters.

    This model is saved to the first available directory index in the 
    parent directory.
    
    Args:
        model (tf.keras.Model) : Model to save.
        history (tf.keras.callbacks.History) : History of training 
            performance parameteres.
        parent_dir_path (str) : Path to parent directory. 
        config (object) : Configuration object used during model 
            training.
        """
    index = 0
    # Search for next available directory
    while True:
        dir_path = os.path.join(parent_dir_path, str(index))
        if not os.path.exists(dir_path):
            break
        index += 1
    os.mkdir(dir_path)
    # Save model
    model_save_path = os.path.join(dir_path, "model")
    model.save(model_save_path)
    # Save Config 
    config_save_path = os.path.join(dir_path, "config.json")
    write_class_to_disk(config, "config", config_save_path)
    # Save history
    history_save_path = os.path.join(dir_path, "history.json")
    write_dict_to_disk(history.history, history_save_path)