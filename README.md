# Required dependencies
* Tensorflow version 2.11.0 (note this is the EXACT version that we have confimed works, newer versions may work but this is NOT guaranteed
* numpy verion 1.24.2 or newer
* matplotlib version 3.7.0 or newer
* seabourn version 0.12.2 or newer
* pillow version 9.4.0

Additionally jupyter notebook functionality is required

# Structure of this repository
All code is located in the "Python" directory which contains 3 jupyter notebooks, a config scrip and a three directories containing the backend functionality.

Each of the jupyter notebooks provides a simple frontend for interfacing with the training logic. The first notebook is called "data_tools" and provides the necessary functionality to convert segmentation masks from RGB encoding the encoding scheme of choice. The second notebook is called "train_model" and provides all the functionality to train new models, improve old models, and view their predictions. The final notebook is called "view_model" and contains the functionality to benchmark trained models and view their predictions.