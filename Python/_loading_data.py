import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

image_dir = os.path.join('Random','data','33', 'JPEGImages')
mask_dir = os.path.join('Random','data','33', 'SegmentationClass')


img_lookup = os.path.join(image_dir, '*.png')
image_names = glob.glob(img_lookup)
image_names.sort()
images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in image_names]
image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis=3)

mask_lookup = os.path.join(mask_dir, '*.png')
mask_names = glob.glob(mask_lookup)
mask_names.sort()
# mask_names_subset = mask_names[0:num_images]
masks = [cv2.imread(mask, cv2.IMREAD_COLOR) for mask in mask_names] #_subset]
mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset, axis = 3)


print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))
# FeaturesDict({
#     'file_name': Text(shape=(), dtype=string),
#     'image': Image(shape=(1024, 1024, 4), dtype=uint8),
#     'label': ClassLabel(shape=(), dtype=int64, num_classes=10),
#     'segmentation_mask': Image(shape=(None, None, 1), dtype=uint8),
# })

