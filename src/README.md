## Instructions for replication

This directory contains all the source code needed to reproduce this work.

### Data preparation

To directly run the train and validation script in this directory, data preparation needs to be done. You can access all the training and testing data via [**datasetCPICANN**](https://huggingface.co/datasets/caobin/datasetCPICANN), the pretrained models via [pretrainCPICANN](https://huggingface.co/caobin/pretrainCPICANN)

File **.pth is the pretrained model, place them under directory "pretrained".

Place directory "train" and "val" from dataset.zip under directory "data", place the annotation files anno_train.csv and anno_val.csv under directory "annotation".

### Model Trianing

#### Single-phase

Run ```python train_single-phase.py``` to train the single-phase identification model from scratch. To train the model on your data, addtional parameters need to be set: ```python train_single-phase.py --data_dir_train=[your training data] --data_dir_val=[your validation data] --anno_train=[your anno file for training data] --anno_val=[your anno file for validation data]```.

#### Bi-phase

Run ```python train_bi-phase.py``` to train the bi-phase identification model. The bi-phase identification model is trained based on single-phase model, you can change the default setting by set the parameter ```load_path=[your pretrained single-phase model]```.

### Model validation

Run ```python train_single-phase.py``` and ```python val_bi-phase.py``` to run the validation code at default setting.

If you wish to validate the model on your data, plase format your data using data_format.py
