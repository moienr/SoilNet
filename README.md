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
The files `self_supervised_train.ipynb` and `train.ipynb` are used for training the self-supervised phase and the final fine-tuning, respectively. You can configure various settings, such as the desired backbone and other hyperparameters, directly within these notebooks. The training process is automatically visualized for you. Finally, you can evaluate your model using the `Accuracy_assessment.ipynb` notebook.

This repository will be updated gradually. Meanwhile, do not hesitate to contact us via: nkakhani@gmail.com and 
moi3nr@gmail.com 