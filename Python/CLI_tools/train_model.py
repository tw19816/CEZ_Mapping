import os
import sys
import tensorflow as tf
from Python.segmentation_model import deeplabv3plus
from Python.config import Config
from Python.utils import write_model_to_disk

if __name__ == "__main__":
    # N_ARGS = 2
    # if (n_args := len(sys.argv)) != N_ARGS:
    #     print(f"Useage: python {sys.argv[0]} <TFDataset number>")
    #     exit()
    # tfdataset_path = os.path.join(Config.tfdata_path, sys.argv[1])
    tfdataset_path = os.path.join(Config.tfdata_path, str(0))
    if not os.path.exists(tfdataset_path):
        raise ValueError(f"Unable to find TFDataset at {tfdataset_path}")
    train_path, val_path, test_path = [
        os.path.join(tfdataset_path, path) for path in ("train", "validation", "test")
    ]
    train_dataset = tf.data.Dataset.load(train_path)
    val_dataset = tf.data.Dataset.load(val_path)
    test_dataset = tf.data.Dataset.load(test_path)
    val_dataset = val_dataset.map(lambda x, y, z :  (x, y))
    model = deeplabv3plus(
        Config.input_shape, 
        Config.batch_size, 
        Config.output_channels,
        Config.channels_low,
        Config.channels_high,
        Config.middle_repeat
    )
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model_history = model.fit(
        train_dataset.prefetch(tf.data.AUTOTUNE),
        epochs=Config.epochs,
        validation_data=val_dataset,
        shuffle=True
    )
    write_model_to_disk(model, model_history, Config)