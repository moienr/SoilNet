# SoilNet
A Spatio-temporal Framework for Soil Property Prediction with Digital Soil Mapping (DSM)

This new architecture, incorporates spatial information using a base convolutional neural network (CNN) model and spatial attention mechanism, along with climate temporal information using a long short-term memory (LSTM) network. 

<!-- ## Experiments

| USE_SA | USE_LSTM | USE_SRTM | OC_MAX | **RUN NAME** |
|  :---:  |  :---:   |  :---:   |  :---:   |    :---:    |
|   ✅   |    ❌   |    ✅    |   87   |      RUN_D_2023_05_04_T_13_27_Moien        |
|   ✅   |    ❌   |    ✅    |   87   |      RUN_D_2023_05_08_T_14_17_Nafiseh  |
|        |          |          |        |              |
|        |          |          |        |              |
|        |          |          |        |              |
|        |          |          |        |              |
|        |          |          |        |              |
|        |          |          |        |              |
|        |          |          |        |              |
|        |          |          |        |              | -->



<!-- ### MODEL
- ~~Add Spatial Attention Module~~
- ~~FCNN + Regressor~~
- ~~FCNN + SA + Regressor~~
- ~~LSTM~~
 -->
 
<!-- ### DATASET
- ~~Add TerraClimate Dataset~~
- ~~Update the ClimateInformation.js (in processing)~~
- ~~Write SRTM + Slope dataset donwlaoder~~ 
 ### Analysis
- CNN
- CNN + Att 
- CNN + Att + LSTM
- RF with no timeseries data
- RF + timeseries data -->

---
<!-- ![oc_all](https://github.com/moienr/SoilNet/blob/d0255c1ce411e631265daf311f1ca0d68b7b0412/readme_imgs/overallarc2.png) -->
---
 ![SoilNET](./readme_imgs/overallarc2.png)
---


# Usage

## Installation

1. **Prerequisites**:
   - Install Conda: Make sure you have Conda installed on your system.

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/moienr/SoilNet.git
   ```

3. **Create the Environments**:
   ```bash
    cd SoilNet
    ```
    3.1. for training:
    ```bash  
    conda env create -f requirements/pytorch_reqs.yml
    ```

    3.2. for dataset:
    ```bash  
    conda env create -f requirements/geemap_reqs.yml
    ```

4. **Activate the Environment**:

    4.1. for train:

    ```bash
    conda activate pytorch
    ```

    4.2. to download the dataset:

    ```bash
    conda activate geemap
    ```

## Training

5. **Run the Code**:

    Flags are explained in the next section.

    ```bash
    python train.py -ne 100 -tbs 8 -ne 10 -ca resnet101
    ```
    
    Although to train, you're gonna need to have the `.csv` files. namely LUCAS dataset under the flag of `--lucas_csv` and the TerraClimate dataset under the flag of `--climate_csv_folder_path`.

The output is a **Training Plot and** a **JSON** file containing all of the results of the **cross-validation**. all will be saved in the `results/` folder







## Explanation of Command-line Flags for `train.py`

The `train.py` script accepts several command-line flags (arguments) that allow you to customize the training process. These flags help adjust various settings and parameters for the model training. Below is a detailed explanation of each flag:

1. `-nw` or `--num_workers`:
   - Type: Integer
   - Default: 2
   - Description: Number of workers used for data loading during training. Adjust this value based on your system's capabilities to optimize data loading efficiency.

2. `-tbs` or `--train_batch_size`:
   - Type: Integer
   - Default: 4
   - Description: Batch size used during training. A larger batch size can increase training speed but may require more memory.

3. `-Tbs` or `--test_batch_size`:
   - Type: Integer
   - Default: 4
   - Description: Batch size used during testing. Similar to the training batch size, it affects memory consumption during testing.

4. `-lr` or `--learning_rate`:
   - Type: Float
   - Default: 0.0001
   - Description: Learning rate used for the optimization algorithm during training.

5. `-ne` or `--num_epochs`:
   - Type: Integer
   - Default: 2
   - Description: Number of training epochs. An epoch is a complete pass through the entire training dataset.

6. `-lrs` or `--lr_scheduler`:
   - Choices: 'step', 'plateau', or None (case-sensitive)
   - Default: 'step'
   - Description: Learning rate scheduler type. Choose from 'step' (step-wise decay), 'plateau' (decay on validation loss plateau), or None (no learning rate decay).

7. `-oc` or `--oc_max`:
   - Type: Integer
   - Default: 87
   - Description: Maximum value for OC during training. Adjust this value based on your specific use case.

8. `-us` or `--use_srtm`:
   - Action: Store True
   - Default: True
   - Description: Enable or disable the use of SRTM (Shuttle Radar Topography Mission) data during training.

9. `-usa` or `--use_spatial_attention`:
   - Action: Store True
   - Default: True
   - Description: Enable or disable the use of spatial attention in the model architecture.

10. `-ca` or `--cnn_architecture`:
    - Choices: 'vgg16', 'resnet101' (case-sensitive)
    - Default: 'vgg16'
    - Description: Choose the CNN architecture for the model. Options are 'vgg16' or 'resnet101'.

11. `-rv` or `--reg_version`:
    - Type: Integer
    - Default: 2
    - Description: Regression version used during training. Adjust this value based on your specific use case.

12. `-ulb` or `--use_lstm_branch`:
    - Action: Store True
    - Default: True
    - Description: Enable or disable the use of the LSTM branch in the model architecture.

13. `-tl8` or `--train_l8_folder_path`:
    - Type: String
    - Default: 'D:\python\SoilNet\dataset\l8_images\\train\\'
    - Description: Path to the training L8 (Landsat 8) image folder.

14. `-tsl8` or `--test_l8_folder_path`:
    - Type: String
    - Default: 'D:\python\SoilNet\dataset\l8_images\\test\\'
    - Description: Path to the test L8 (Landsat 8) image folder.

15. `-vl8` or `--val_l8_folder_path`:
    - Type: String
    - Default: 'D:\python\SoilNet\dataset\l8_images\\val\\'
    - Description: Path to the validation L8 (Landsat 8) image folder.

16. `-tvsl8` or `--testval_l8_folder_path`:
    - Type: String
    - Default: 'D:\python\SoilNet\dataset\l8_images\\val\\'
    - Description: Path to the test/validation L8 (Landsat 8) image folder.

17. `-lcp` or `--lucas_csv_path`:
    - Type: String
    - Default: 'D:\python\SoilNet\dataset\LUCAS_2015_all.csv'
    - Description: Path to the LUCAS CSV file.

18. `-ccp` or `--climate_csv_folder_path`:
    - Type: String
    - Default: "D:\\python\\SoilNet\\dataset\\Climate\\All\\filled\\"
    - Description: Path to the climate CSV folder.

These command-line flags allow you to configure various aspects of the model training process based on your specific requirements and dataset. Adjust the values according to your needs when running the `train.py` script.
