import tensorflow as tf

def dataset_setup(directory: str, batch: int, val_split: float):
    '''Creates the necessary datasets with training and validation split val_split.
    Args:
        directory (str) : Directory of dataset.
        batch_size (int) : Number of inputs per batch.
        val_split (float) :  Float between 0 and 1, fraction of data to reserve for validation.

    Returns:
        train_dataset (tf.data.Dataset) : Tf.data.Dataset object containing training dataset.
        val_dataset (tf.data.Dataset) : Tf.data.Dataset object containing validation dataset.
    '''

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None,
        color_mode='rgb',
        batch_size=batch,
        image_size=(1024, 1024),
        shuffle=False,
        validation_split=val_split,
        subset='training',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None,
        color_mode='rgb',
        batch_size=batch,
        image_size=(1024, 1024),
        shuffle=False,
        validation_split=val_split,
        subset='validation',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        )
    return train_dataset, val_dataset

def normalise(dataset, pixels: int = 255):
    '''Normalises pixel values to float range 0 and 1.
    Args:
        dataset (tf.data.Dataset): TensorFlow dataset object with images encoded.
        pixels (int): Max pixel value.
    Returns:
        normalised_ds (tf.data.Dataset): TensorFlow dataset object with rescaled pixel value images encoded.
    '''

    normalisation_layer = tf.keras.layers.Rescaling(1./pixels)
    normalised_ds = dataset.map(lambda x: (normalisation_layer(x)))
    return normalised_ds

def data_pipeline(directory: str, batch: int, val_split: float, pixels: int = 255):
   '''Loads the image data and segmentation masks into a tf.data.Dataset object. 
    It splits the data into the training set and validation set with a validation split (val_split).
    Then normalises the image data.
    Args:
        directory (str) : Directory of dataset. (NOTE: Image files must be in JPEGImages and masks in SegmentationClass.)
        batch_size (int) : Number of inputs per batch.
        val_split (float) :  Float between 0 and 1, fraction of data to reserve for validation.
        pixels (int): Max pixel value.

    Returns:
        norm_train_dataset (tf.data.Dataset) : Tf.data.Dataset object containing training dataset.
        norm_val_dataset (tf.data.Dataset) : Tf.data.Dataset object containing validation dataset.
        mask
        '''
   print('NOT COMPLETE')


