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

1. **Prerequisites**:
   - Install Conda: Make sure you have Conda installed on your system.
   - Internet Connection: Ensure you have an active internet connection.

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/moienr/SoilNet.git
   ```

3. **Create the Environments**:
   ```bash
    cd SoilNet
    ```
    3.1. for train:
    ```bash  
    conda env create -f requirements/pytorch_reqs.yml
    ```

    3.2. for dataset:
    ```bash  
    conda env create -f requirements/geemap_reqs.yml
    ```

4. **Activate the Environment**:
    ```bash
    conda activate pytorch
    ```
    or
    ```bash
    conda activate geemap
    ```


5. **Run the Code**:

    ```bash
    python train.py -nw 4 -tbs 8 -lr 0.0001 -ne 10 -ca resnet101
    ```
    Although to train, you're gonna need to have the `.csv` files. namely LUCAS dataset under the handle of `--lucas_csv` and the TerraClimate dataset under the handle of `--climate_csv_folder_path`. 
