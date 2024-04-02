# iORDeepPred

## Introduction
iORDeepPred is a deep learning model designed for predicting the functionality of insect olfactory receptors. It enables rapid and efficient batch prediction of the interaction between insect olfactory receptors and VOCs (volatile organic compounds). We provide a pre-trained model on large-scale molecular docking data, allowing users to perform transfer learning based on existing experimental data to obtain efficient models for predicting the functionality of insect olfactory receptors. For more detailed information on iORDeepPred, please refer to: "**Functional Prediction of Insect Olfactory Receptors Using Multi-transfer Learning from Virtual Screening to Experiments**" (to be published).

## Dependence

`h5py==3.6.0`

`numpy==1.21.5`

`pandas==1.3.5`

`pytorch_lightning==1.6.4`

`scikit_learn==1.4.1.post1`

`torch==1.11.0`

`tqdm==4.64.0`

`transformers==4.19.2`


## Project Structure
**csv_file**: CSV format file for input
**output**: In the data preprocessing process, store intermediate files
**protT5**: Pretrained models used for OR feature extraction storage
**Smodel**: Pretrained models used for VOC feature extraction storage
**space**: Store trained models
**all_script**: The relevant functions for data preprocessing
**config.py**: The configuration file of the model
**input_data.npy**: The processed data file
**iORDeepPred.py**: The main file
**model.py**: The model definition file
**model_test.py**: The model definition file
**requirements.txt**: The environment dependency file
**test.csv**: The predicted results
**test.py**: The functions related to the prediction module
**train.py**: The functions related to the training module
**train_test.py**: The relevant functions
**utils.py**: The relevant functions
**utils_test.py**: The relevant functions

Tips：The pretrained model file for feature extraction is too large. Currently, users need to download it themselves from [hugging_face](http://www.huggingface.co) and place it in the protT5 and Smodel directories.

## Usage
### 1.Parameter configuration
Before proceeding with further functional predictions, you can refer to the functional help documentation of iORDeepPred:

`python iORDeepPred.py -h`

You can review the specific meanings of each parameter, and then users can add corresponding parameters when executing subsequent commands.

### 2.Train the model using experimental data
（1）Organize the experimental data into CSV format files, where OR, VOC, and their interaction relationships are saved as seq.csv, voc.csv, and inter.csv respectively. Please refer to the example files in the csv_file directory for the specific format of CSV files.
（2）Preprocess the experimental data to transform it into a format suitable for model reading. Use the data preprocessing module of iORDeepPred:

`python iORDeepPred.py -d`

（3）Train the model using the processed data. Utilize the model training module of iORDeepPred:

`python iORDeepPred.py -t`


### 3.Utilize the trained model to predict the interaction relationships for the target OR-VOC pairs
（1）Input data processing. All operations are consistent with steps (1) and (2) in the previous training model. It is worth noting that whether for training or prediction, the format of the input data files is exactly the same. The difference lies in: during training, inter.csv needs to contain OR-VOC pairs and the strength of their interactions; during prediction, the inter.csv file only provides the OR-VOC pairs that need to be predicted.
（2）Predict the target OR-VOC pairs using the functional prediction module of iORDeepPred：

`python iORDeepPred.py -p`



