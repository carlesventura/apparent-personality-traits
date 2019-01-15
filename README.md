# Interpreting CNN Models for Apparent Personality Trait Regression

This repository refers to the paper **Interpreting CNN Models for Apparent Personality Trait Regression** that can be found in [this link](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w26/papers/Ventura_Interpreting_CNN_Models_CVPR_2017_paper.pdf).

The code should be used in the following order:
- Using the whole image:
    1. Training the model: train/train_regression_DANplus_images.m
    2. Prediction code over test: val/eval_val_reg_avg_max_l28_images.m
    3. Evaluation of the model: val/compute_accuracy.m
    4. Training the model for interpretability: train/train_regression_DANplus_images_localization.m
    5. Prediction code over test for model trained for intrepratibility: val/eval_val_reg_avg_max_l28_images.m
    6. Code for finding the images with maximum activation for each personality trait: val/find_highest_activation_images.m
    7. Code for generating the images with maximum activation for each personality trait with the discriminative localizations: val/localization_highest_images.m
    8. Code for finding the K videos with highest activation for a specific unit from a given layer and obtaining the image segmentation: val/activation_unit_highest_image.m
- Using the cropped face images:
    1. Training the model: train/train_regression_DANplus_faces.m
    2. Prediction code over test: val/eval_val_reg_avg_max_l28_faces.m
    3. Evaluation of the model: val/compute_accuracy.m
    4. Training the model for interpretability: train/train_regression_DANplus_faces_localization.m
    5. Prediction code over test for model trained for intrepratibility: val/eval_val_reg_avg_max_l28_faces_localization.m
    6. Code for finding the images with maximum activation for each personality trait: val/find_highest_activation_images_faces.m
    7. Code for generating the images with maximum activation for each personality trait with the discriminative localizations: val/localization_highest_faces.m
    8. Code for finding the K videos with highest activation for a specific unit from a given layer and obtaining the image segmentation: val/activation_unit_highest_faces.m
