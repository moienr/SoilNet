from train_SimCLR_utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import numpy as np
from dataset.utils.utils import TextColors as tc
from plot_utils.plot import plot_train_test_losses
from datetime import date, datetime
import torch.nn.functional as F
import cv2
import json

import os
from soilnet.soil_net import SoilNet, SoilNetLSTM, SoilNetSimCLR

from datetime import date, datetime
import config
import argparse


# create a folder called 'results' in the current directory if it doesn't exist
if not os.path.exists('results'):
    os.mkdir('results')
    
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
EXP_NAME = 'LUCAS_Self560_ViT_Trans'
NUM_WORKERS = 2
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS =  2
LR_SCHEDULER = "step" # step, plateau or None
DATASET = 'LUCAS' # 'LUCAS', 'RaCA'
USE_SRTM = False
USE_SPATIAL_ATTENTION =  False
CNN_ARCHITECTURE = 'ViT' # vgg16 or resnet101 or "ViT"
RNN_ARCHITECTURE = 'Transformer' # LSTM, GRU, RNN, Transformer
REG_VERSION = 1 # Regression version 1 or 2
USE_LSTM_BRANCH = False # If True, the model will use the climate data
SEEDS = [1,] # seeds for the cross validation
def parse_arguments():
    parser = argparse.ArgumentParser(description='SoilNet SSL Training')
    parser.add_argument('-exp', '--experiment_name', type=str, default=EXP_NAME, help='Experiment name')
    parser.add_argument('-nw', '--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')
    parser.add_argument('-trbs', '--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('-tsbs', '--test_batch_size', type=int, default=TEST_BATCH_SIZE, help='Batch size for testing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-ne', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-lrs', '--lr_scheduler', type=str, default=LR_SCHEDULER, choices=['step', 'plateau', 'None'], help='Learning rate scheduler')
    parser.add_argument('-ds', '--dataset', type=str, default=DATASET, choices=['LUCAS', 'RaCA'], help='Dataset name')
    parser.add_argument('-srtm', '--use_srtm', action='store_true', default=USE_SRTM, help='Use SRTM data')
    parser.add_argument('-sa', '--use_spatial_attention', action='store_true', default=USE_SPATIAL_ATTENTION, help='Use spatial attention')
    parser.add_argument('-cnn', '--cnn_architecture', type=str, default=CNN_ARCHITECTURE, choices=['vgg16', 'resnet101', 'ViT'], help='CNN architecture')
    parser.add_argument('-rnn', '--rnn_architecture', type=str, default=RNN_ARCHITECTURE, choices=['LSTM', 'GRU', 'RNN', 'Transformer'], help='RNN architecture')
    parser.add_argument('-rv', '--reg_version', type=int, default=REG_VERSION, choices=[1, 2], help='Regression version')
    parser.add_argument('-lstm', '--use_lstm_branch', action='store_true', default=USE_LSTM_BRANCH, help='Use Climate data (I know! the name is misleading!)') 
    parser.add_argument('-s', '--seeds', nargs='+', type=int, default=SEEDS, help='Seeds for cross validation')
    
    return parser.parse_args()


train_l8_folder_path = config.train_l8_folder_path
test_l8_folder_path = config.test_l8_folder_path
val_l8_folder_path = config.val_l8_folder_path
lucas_csv_path = config.lucas_csv_path
climate_csv_folder_path = config.climate_csv_folder_path


if __name__ == '__main__':
    # Format the date and time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current Date and Time:", start_string)

    parser = parse_arguments()
    args = parser.parse_args()
    EXP_NAME = args.experiment_name
    NUM_WORKERS = args.num_workers
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    LR_SCHEDULER = args.lr_scheduler
    DATASET = args.dataset
    USE_SRTM = args.use_srtm
    USE_SPATIAL_ATTENTION = args.use_spatial_attention
    CNN_ARCHITECTURE = args.cnn_architecture
    RNN_ARCHITECTURE = args.rnn_architecture
    REG_VERSION = args.reg_version
    USE_LSTM_BRANCH = args.use_lstm_branch
    SEEDS = args.seeds

    if DATASET == 'LUCAS':
        from dataset.dataset_loader import SNDataset,SNDatasetClimate, myNormalize, myToTensor, Augmentations
        OC_MAX = 560.2
    elif DATASET == 'RaCA':
        from dataset.dataset_loader_us import SNDataset,SNDatasetClimate, myNormalize, myToTensor, Augmentations
        OC_MAX = 4115
    else:
        raise ValueError('Invalid dataset Name')

    if USE_SRTM:
        mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], oc_min = 0, oc_max = OC_MAX)
    else:
        mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)]], oc_min = 0, oc_max = OC_MAX)
        
    my_to_tensor = myToTensor()
    my_augmentation = Augmentations()
    train_transform = transforms.Compose([mynorm, my_to_tensor,my_augmentation])
    test_transform = transforms.Compose([mynorm, my_to_tensor])

    bands = [0,1,2,3,4,5,6,7,8,9,10,11] if not USE_SRTM else [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

    ################################# IF Not USE_LSTM_BRANCH ###############################
    if not USE_LSTM_BRANCH: # NOT USING THE CLIMATE DATA

        train_ds = SNDataset(train_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=train_transform)
        test_ds =  SNDataset(test_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=test_transform)
        val_ds = SNDataset(val_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=test_transform)
        test_ds_w_id =  SNDataset(test_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=test_transform, return_point_id=True)

        
    ################################### IF USE_LSTM_BRANCH #################################
    else: # USING THE CLIMATE DATA
        train_ds = SNDatasetClimate(train_l8_folder_path,
                                        lucas_csv_path,
                                        climate_csv_folder_path,
                                        l8_bands=bands, transform=train_transform)

        test_ds = SNDatasetClimate(test_l8_folder_path,
                                    lucas_csv_path,
                                    climate_csv_folder_path,
                                    l8_bands=bands, transform=test_transform)
        
        val_ds = SNDatasetClimate(val_l8_folder_path,
                                    lucas_csv_path,
                                    climate_csv_folder_path,
                                    l8_bands=bands, transform=test_transform)
        
        test_ds_w_id = SNDatasetClimate(test_l8_folder_path,
                                    lucas_csv_path,
                                    climate_csv_folder_path,
                                    l8_bands=bands, transform=test_transform, return_point_id=True)

        
    SEQ_LEN = test_ds_w_id[0][0][1].shape[0]



    # COUNTING the csv files in the csv folder
    CSV_FILES = [f for f in os.listdir(climate_csv_folder_path) if f.endswith('.csv')]
    NUM_CLIMATE_FEATURES = len(CSV_FILES)


    cv_results = {"train_loss": [],
                    "train_acc_top1": [],
                    "train_acc_top5": [],
                    "train_acc_mean_pos": [],
                    "val_loss": [],
                    "val_acc_top1": [],
                    "val_acc_top5": [],
                    "val_acc_mean_pos": [],
        }


    # Format the date and time
    now = datetime.now()
    run_name = now.strftime("D_%Y_%m_%d_T_%H_%M")
    print("Current Date and Time:", run_name)
    # create a folder called 'results' in the current directory if it doesn't exist
    if not os.path.exists('results'):
        os.mkdir('results')
        

    best_mae = 1000 # just a big number, since our data is normalized between 0 and 1, mae is between 0 and 1 too.
    best_seed = SEEDS[0]
    for idx, seed in enumerate(SEEDS):
        print(tc.BOLD_BAKGROUNDs.PURPLE, f"CROSS VAL {idx+1}", tc.ENDC)
        
        
        train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        val_dl = DataLoader(val_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        #model = SoilNetFC(cnn_in_channels=12, regresor_input_from_cnn=1024, hidden_size=128).to(device)
        # architecture = "101+GLAM" if USE_SPATIAL_ATTENTION else "101"
        if USE_LSTM_BRANCH:
            model = SoilNetSimCLR(use_glam=USE_SPATIAL_ATTENTION, cnn_arch= CNN_ARCHITECTURE, reg_version= REG_VERSION,
                            cnn_in_channels=len(bands), regresor_input_from_cnn=128,
                            lstm_n_features= NUM_CLIMATE_FEATURES, lstm_n_layers= 2, lstm_out= 128,
                            hidden_size=128, rnn_arch=RNN_ARCHITECTURE,seq_len=SEQ_LEN).to(device)
        else:
            model = SoilNet(use_glam=USE_SPATIAL_ATTENTION, cnn_arch = CNN_ARCHITECTURE, reg_version= REG_VERSION,
                        cnn_in_channels=len(bands), regresor_input_from_cnn=128, hidden_size=128).to(device)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Saving the model on the last epoch
        save_model_path = f"results/RUN_{EXP_NAME}_{run_name}.pth.tar"
        
        results = train(model, train_dl, test_dl, val_dl,
                        torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
                        SimCLR(temperature=0.5), epochs=NUM_EPOCHS, lr_scheduler=LR_SCHEDULER,
                        save_model_path= save_model_path,
                        save_model_if_mae_lower_than= best_mae,
                        )

        
        cv_results['train_loss'].append(results['train_loss'])
        cv_results['train_acc_top1'].append(results['train_acc_top1'])
        cv_results['train_acc_top5'].append(results['train_acc_top5'])
        cv_results['train_acc_mean_pos'].append(results['train_acc_mean_pos'])
        cv_results['val_loss'].append(results['val_loss'])
        cv_results['val_acc_top1'].append(results['val_acc_top1'])
        cv_results['val_acc_top5'].append(results['val_acc_top5'])
        cv_results['val_acc_mean_pos'].append(results['val_acc_mean_pos'])
        
        
        
    torch.save(model, f"results/RUN_{EXP_NAME}_{run_name}_SelfSupervised.pth")

 
    train_arr = np.asarray(cv_results['train_acc_mean_pos'])
    val_arr = np.asarray(cv_results['val_acc_mean_pos'])
    train_arr.shape, val_arr.shape
    plot_train_test_losses(train_arr,val_arr, title="Average Self-Rank", x_label="Epochs", y_label="Rank",
                        min_max_bounds= True, tight_x_lim= True,
                        train_legend = "Train", test_legend = "Validation",
                        save_path=f"results/RUN_{run_name}.png",show=False)


    train_arr = np.asarray(cv_results['train_acc_top5'])
    val_arr = np.asarray(cv_results['val_acc_top5'])
    train_arr.shape, val_arr.shape
    plot_train_test_losses(train_arr,val_arr, title="Top 5 Prob", x_label="Epochs", y_label="Top 5 Prob",
                        min_max_bounds= True, tight_x_lim= True,
                        train_legend = "Train", test_legend = "Validation",
                        save_path=f"results/RUN_{EXP_NAME}_{run_name}.png",show=False)


    train_arr = np.asarray(cv_results['train_acc_top1'])
    val_arr = np.asarray(cv_results['val_acc_top1'])
    train_arr.shape, val_arr.shape
    plot_train_test_losses(train_arr,val_arr, title="Top 1 Prob", x_label="Epochs", y_label="Top 1 Prob",
                        min_max_bounds= True, tight_x_lim= True,
                        train_legend = "Train", test_legend = "Validation",
                        save_path=f"results/RUN_{EXP_NAME}_{run_name}.png",show=False)
    plt.show()    
    
    print(tc.BOLD_BAKGROUNDs.GREEN, "TRAINING FINISHED", tc.ENDC)