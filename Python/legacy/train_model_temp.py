from Python.segmentation_model import deeplabv3plus
from Python.config import Config
from Python import load_numpy as load
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    image_dir = r"/home/vidarmarsh/CEZ_Mapping/Data/33/JPEGImages"
    greyscale_dir = r"/home/vidarmarsh/CEZ_Mapping/Data/33/SegmentationClass_greyscale"
    image_dataset, image_paths = load.load_image_dir_to_array(
        image_dir, sorted=True, rgb=True
    )
    old_shape = image_dataset.shape
    new_shape = (*old_shape[:-2], old_shape[-1])
    image_dataset = image_dataset.reshape(new_shape)
    # image_dataset = np.expand_dims(image_dataset, axis=0)
    seg_dataset, seg_paths = load.load_image_dir_to_array(
        greyscale_dir, sorted=True, rgb=False
    )
    # seg_dataset = np.expand_dims(seg_dataset, axis=0)

    # seg_dataset = seg_dataset.reshape(new_shape)
    full_dataset = tf.data.Dataset.from_tensor_slices(
        (image_dataset, seg_dataset), name="Dataset"
    )
    full_dataset = full_dataset.batch(1)
    print(full_dataset.cardinality())
    n_images = len(full_dataset)
    train_size = int(n_images * 0.7)
    val_size = int(n_images * 0.15)
    test_size = int(n_images * 0.15)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    model = deeplabv3plus((1024, 1024, 3), None, 11, channels_high=256, middle_repeat=0)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model_history = model.fit(
        train_dataset, epochs=20,
        validation_data=val_dataset
        )


