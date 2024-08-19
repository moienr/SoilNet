# SoilNet

A Hybrid Transformer-based Framework with Self-Supervised Learning for Large-scale Soil Organic Carbon Prediction has been presented - Accepted for publication in IEEE Transactions on Geoscience and Remote Sensing (TGRS). 
The training consists of two phases: 

1) Self-supervised contrastive learning and <br>
2) Supervised fine-tuning via ground truth for our downstream task, which is regression.

---
![Graohical_abstract](https://github.com/moienr/SoilNet/blob/main/readme_imgs/Graphical_abs2.jpg)
---
## Usage

### Setup Instructions

First, you need to clone the repository to your local machine. You can do this using the following command:

```bash
git clone https://github.com/moienr/SoilNet
```

Next, you should create two Conda environments: one for data preparation and acquisition using Google Earth Engine, and the other for the deep learning model. The environment configurations can be found in the requirements folder.

You can create the environments using the following command:

```bash
conda env create -f <environment_file.yml>
```

Replace `<environment_file.yml>` with the appropriate file name from the `requirements` folder.

This study utilizes two types of data: raster-based data, which includes remote sensing images, indices, and topographical information, and time series data, which encompasses climate variables.<br>
During the initial phase of training, which is self-supervised, ground references are not required, as the process involves random locations. However, for the fine-tuning phase, ground truth data is essential. You can run `L8_dataset_downloader.ipynb` to prepare your data.

### Ground Truth
Our model has been trained via two different datasets:
- LUCAS: To access the LUCAS topsoil dataset (ground truth), visit: [LUCAS Topsoil Dataset](https://esdac.jrc.ec.europa.eu/content/topsoil-physical-properties-europe-based-lucas-topsoil-data)
- RaCA: To access the RaCA dataset (ground truth), visit: [RaCA Dataset](https://www.nrcs.usda.gov/resources/data-and-reports/rapid-carbon-assessment-raca)

### Model Training 
Training the model consists of two phases: self-supervised learning and supervised fine-tuning. The files `train_ssl.py` and `train.py` are used for training the self-supervised phase and the final fine-tuning, respectively.

Using the following instructions, you can train your own model with any image feature extractor backbone, and Time Series architecture backbone. 

**0. Data Preparation:**

Set the paths to the data in the `config.py` file.
```python
train_l8_folder_path = '/dataset/l8_images/train/'
test_l8_folder_path = '/dataset/l8_images/test/'
val_l8_folder_path = '/dataset/l8_images/val/'
lucas_csv_path = '/dataset/LUCAS_2015_all.csv'
climate_csv_folder_path = "/dataset/Climate/All/filled/"
# if have self-supervised pre-trained model:
SIMCLR_PATH = "/project/results/RUN_LUCAS_Self560_ViT_Trans_D_2024_08_19_T_16_13_SelfSupervised.pth"
```

**1. Self-supervised learning:**

*(This phase can be omitted if you have access to the pre-trained model or you want to only use supervised learning)*

To train the model in the self-supervised phase, you can run `train_ssl.py` with the following command:

```bash
python train_ssl.py --num_workers 8 --trbs 64 --lr 0.0001 --num_epochs 100 --lr_scheduler 'step' --dataset 'LUCAS' --use_srtm --use_lstm_branch --cnn_architecture 'ViT' --rnn_architecture 'Transformer' --seeds 1 42 86

```

Based on the experiment name, and the time of running the script, the results and pre-trained model will be saved in the `results` folder.

You should add the paths to this saved model in the `config.py` file to use it in the fine-tuning phase.


**2. Fine-tuning:**

To fine-tuning the pre-trained model, you can run `train.py` with the following command:

```bash
python train.py --dataset 'LUCAS' --num_workers 8 --load_simclr_model --trbs 64 --lr 0.0001 --num_epochs 100 --lr_scheduler 'step' --use_srtm --use_lstm_branch --seeds 1 42 86 

```
*Note:* if you use `--load_simclr_model`, your architecture will be overwritten by the pre-trained model architecture.

Or for training from scratch:

```bash
python train.py --dataset 'LUCAS' --num_workers 8 --cnn_architecture 'ViT' --rnn_architecture 'Transformer' --trbs 64 --lr 0.0001 --num_epochs 100 --lr_scheduler 'step' --use_srtm --use_lstm_branch --seeds 1 42 86 

```






**Help:**

For a detailed explanation of the arguments, you can run the following command:

```bash
python train_ssl.py --help
python train.py --help
```

*If you are a "notebook person", you can use the code in [Release 2.0.0](https://github.com/moienr/SoilNet/releases/tag/v2.0.0) or  [Pre-Release](https://github.com/moienr/SoilNet/tree/ieee-prerelease) branch.*

This repository will be updated gradually. Meanwhile, do not hesitate to contact us via: nkakhani@gmail.com and 
moienrangzan@gmail.com
