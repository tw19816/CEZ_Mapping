import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def load_model(path_to_model: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(path_to_model)
    return model\


def model_predict_from_file(
    model: tf.keras.Model, filepath: str, normalised: bool = False
):
    image = plt.imread(filepath)
    if not normalised:
        image = (image * 255).astype(np.int8)
    dataset = tf.data.Dataset.from_tensors(image)
    mask = model.predict(dataset)
    return mask


if __name__ == "__main__":
    path_to_model = r""
    path_to_image = r""
    model = load_model(path_to_model)
    mask = model_predict_from_file(model, path_to_image)