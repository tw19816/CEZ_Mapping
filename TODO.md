# solution for memory issue
* generate weights for all images and save them to disk
* generate augmentations for all images, masks, and weights and save them to disk
* write a generator that inherits from tensorflow.sequence to batch load images
    * note make segmentation masks

# todo for finalised pipeline
* fix data pipeline DONE
* fix image sizing (512 per pixel) DONE
* fix parameter saving DONE
* fix data visualisation DONE
* solve class inbalance issues DONE (ish)
* make training and validation sets more varied DONE

# Things That Need to be Done:
* Create data pipeline for sythesising a large dataset
    * Function for accepting raw images and annotations for segmentation mask
        * need to deterimine if masks are inputted as bitmaps or coco .json format
    * data normalisation
    * data synthesis
    
* Create model

* Create training pipeline for training model from dataset

* Create pipeline for visualising training results
    * Evolution of:
        * training loss
        * training accuracy
        * validation loss
        * validation accuray
    * Tool for visualising predictions made by model
        * original image
        * originial image overlayed with segmentation mask
        * segmentation mask on black background

