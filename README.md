# Group 7 ISIC 2024 Skin Cancer Detection Model 

The project includes both python (.py) and python notebook (.ipynb) files. Additionally included is the training data. Please follow the directions below and set the needed environmental variables.

![Image](docs/header.png)

https://www.kaggle.com/competitions/isic-2024-challenge
* `train.py`: Script that trains and tests a simple model 
* `isic-2024.ipynb`: Same as train.py, but in a Jupyter Notebook format
## Running `train.py`

#### Using pip

1. Install dependencies
   #### Windows
2. ```bash
   pip install -r requirements.txt
   ```
   #### Linux/MacOS
    ````bash
   pip3 install -r requirments.txt

3. Set enviromental variables (update Data Path)
      #### Windows (Powershell)
      ```bash
      [Environment]::SetEnvironmentVariable('DATA', 'C:\temp', 'Process')
      [Environment]::SetEnvironmentVariable('TRAIN_CSV', "$($env:DATA)\isic-2024-challenge\train-metadata.csv", 'Process')
      [Environment]::SetEnvironmentVariable('TEST_CSV', "$($env:DATA)\isic-2024-challenge\test-metadata.csv", 'Process')
      [Environment]::SetEnvironmentVariable('TRAIN_HDF5', "$($env:DATA)\isic-2024-challenge\train-image.hdf5", 'Process')
      [Environment]::SetEnvironmentVariable('TEST_HDF5', "$($env:DATA)\isic-2024-challenge\dev-image.hdf5", 'Process')
      [Environment]::SetEnvironmentVariable('PRETRAINED_MODEL', "$($env:DATA)\tf_efficientnetv2_b1-be6e41b0.pth", 'Process')
      ````

      #### Linux/MacOS
      ```bash
    export DATA=/Users/bortiz/Desktop
    export TRAIN_CSV=$DATA/isic-2024-challenge/train-metadata.csv
    export TEST_CSV=$DATA/isic-2024-challenge/test-metadata.csv
    export TRAIN_HDF5=$DATA/isic-2024-challenge/train-image.hdf5
    export TEST_HDF5=$DATA/isic-2024-challenge/test-image.hdf5
    export PRETRAINED_MODEL=$DATA/tf_efficientnetv2_b1-be6e41b0.pth
      ```


3. Run the script

   #### Windows
   ```bash
   python train.py
   ```
   #### Linux/MacOS
   ```bash
   python3 train.py
   ```
