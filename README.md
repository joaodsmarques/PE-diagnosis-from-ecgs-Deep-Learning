# Pulmonary embolism diagnosis from Electrocardiograms Deep-Learning

Code of the paper:
### Artificial intelligence-based diagnosis of acute pulmonary embolism: Development of a machine learning model using 12-lead electrocardiogram

### :test_tube: Paper available at:
https://doi.org/10.1016/j.repc.2023.03.016

Using Deep learning to diagnose pulmonary embolism (PE) from ECGs only.


### :warning: Important information:
There are 3 models: 
  - Baseline inspired on the paper https://www.nature.com/articles/s41467-020-15432-4 
  - 1D ResNet-18 enhanced with a multihead self-attention layer
  - 2D ResNet-18 enhanced with a multihead self-attention layer

The first two models receive 12 lead ECG raw data, while the 2D network receives spectrograms. Each lead is composed of 4096 samples with a sampling frequency of 500Hz.

## :books: Dataset
It was trained on an imbalanced dataset retrieved from the Hospital de Santa Maria database containing 1014 examples in total. Due to privacy policies, it is not publicly available.
Each sample is composed of 12 leads with 5000 data points, correspondent to a signal length of 10s (fs = 500 Hz)
The raw data is in xml format, and the sample's leads have the following identifiers ['I','II','III','aVR','aVL','aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'].

## :computer: How to run
There are 3 scripts, each one responsible for the training process of each network:
  - train_atten.py is the main 1D model training script correspondent to the resnet18_with_atten.py model;
  - train_baseline is the baseline model training script correspondent to the res_net_model.py architecture;
  - train_spectrogram is the 2D modeltraining script correspondent to the resnet18_with_atten_spectr.py model;

The files data_augmentation, focal loss and metrics_and_performance contain the data augmentation techniques applied, the focal loss implementation and metric related functions.

you can simply run each model, having all the files in the same location, with the command: 
#### -python 3 script_name.py

Each hyperparameter has to be defined inside each script.

## :writing_hand: Updates:

There are two new models and a single script to run those two added to this repository. 

- The script is called train_main_ecg and it is possible to tune hyperparameters using the YML file called "hyperparameters"
- The RNN model file enables the use of a hybrid neural network composed of a ResNet and a RNN, LSTM or GRU.
- The encoder model makes use of a transformer encoder with traditional positional encoding.

## :shield: License
Project is distributed under MIT License
