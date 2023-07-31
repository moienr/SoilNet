from train_utils import *
import torch
from torch.utils.data import DataLoader
from dataset.dataset_loader import SNDataset,SNDatasetClimate, myNormalize, myToTensor, Augmentations
from torchvision import transforms
import random
import numpy as np
from dataset.utils.utils import TextColors as tc
from plot_utils.plot import plot_train_test_losses
from datetime import date, datetime
import torch.nn.functional as F
import cv2
import json
import argparse


# create a folder called 'results' in the current directory if it doesn't exist
if not os.path.exists('results'):
    os.mkdir('results')
    
    

# Format the date and time
now = datetime.now()
start_string = now.strftime("%Y-%m-%d %H:%M:%S")



import os
os.getcwd()


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


import argparse

# CONFIG
NUM_WORKERS = 2
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
LR_SCHEDULER = "step"  # step, plateau or None
OC_MAX = 87
USE_SRTM = True
USE_SPATIAL_ATTENTION = False
CNN_ARCHITECTURE = 'vgg16'  # vgg16 or resnet101
REG_VERSION = 1
USE_LSTM_BRANCH = False

train_l8_folder_path = 'D:\python\SoilNet\dataset\l8_images\\train\\'
test_l8_folder_path = 'D:\python\SoilNet\dataset\l8_images\\test\\'
val_l8_folder_path = 'D:\python\SoilNet\dataset\l8_images\\val\\'
testval_l8_folder_path = 'D:\python\SoilNet\dataset\l8_images\\val\\'
lucas_csv_path = 'D:\python\SoilNet\dataset\LUCAS_2015_all.csv'
climate_csv_folder_path = "D:\\python\\SoilNet\\dataset\\Climate\\All\\filled\\"

def parse_args():
    parser = argparse.ArgumentParser(description='Train your model.')

    parser.add_argument('-nw', '--num_workers', type=int, default=NUM_WORKERS,
                        help='Number of workers for data loading (default: %(default)s)')
    parser.add_argument('-tbs', '--train_batch_size', type=int, default=TRAIN_BATCH_SIZE,
                        help='Training batch size (default: %(default)s)')
    parser.add_argument('-Tbs', '--test_batch_size', type=int, default=TEST_BATCH_SIZE,
                        help='Test batch size (default: %(default)s)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('-ne', '--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-lrs', '--lr_scheduler', choices=['step', 'plateau', None], default=LR_SCHEDULER,
                        help='Learning rate scheduler type (default: %(default)s)')
    parser.add_argument('-oc', '--oc_max', type=int, default=OC_MAX,
                        help='OC max value (default: %(default)s)')
    parser.add_argument('-us', '--use_srtm', action='store_true', default=USE_SRTM,
                        help='Whether to use SRTM (default: %(default)s)')
    parser.add_argument('-usa', '--use_spatial_attention', action='store_true', default=USE_SPATIAL_ATTENTION,
                        help='Whether to use spatial attention (default: %(default)s)')
    parser.add_argument('-ca', '--cnn_architecture', choices=['vgg16', 'resnet101'], default=CNN_ARCHITECTURE,
                        help='CNN architecture (default: %(default)s)')
    parser.add_argument('-rv', '--reg_version', type=int, default=REG_VERSION,
                        help='Regression version (default: %(default)s)')
    parser.add_argument('-ulb', '--use_lstm_branch', action='store_true', default=USE_LSTM_BRANCH,
                        help='Whether to use LSTM branch (default: %(default)s)')
    parser.add_argument('-tl8', '--train_l8_folder_path', type=str, default=train_l8_folder_path,
                        help='Path to the training L8 folder (default: %(default)s)')
    parser.add_argument('-tsl8', '--test_l8_folder_path', type=str, default=test_l8_folder_path,
                        help='Path to the test L8 folder (default: %(default)s)')
    parser.add_argument('-vl8', '--val_l8_folder_path', type=str, default=val_l8_folder_path,
                        help='Path to the validation L8 folder (default: %(default)s)')
    parser.add_argument('-tvsl8', '--testval_l8_folder_path', type=str, default=testval_l8_folder_path,
                        help='Path to the test/validation L8 folder (default: %(default)s)')
    parser.add_argument('-lcp', '--lucas_csv_path', type=str, default=lucas_csv_path,
                        help='Path to the LUCAS CSV file (default: %(default)s)')
    parser.add_argument('-ccp', '--climate_csv_folder_path', type=str, default=climate_csv_folder_path,
                        help='Path to the climate CSV folder (default: %(default)s)')

    return parser.parse_args()

if __name__ == '__main__':
    print("main "+__name__)
    args = parse_args()
    NUM_WORKERS = args.num_workers
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    LR_SCHEDULER = args.lr_scheduler
    OC_MAX = args.oc_max
    USE_SRTM = args.use_srtm
    USE_SPATIAL_ATTENTION = args.use_spatial_attention
    CNN_ARCHITECTURE = args.cnn_architecture
    REG_VERSION = args.reg_version
    USE_LSTM_BRANCH = args.use_lstm_branch
    train_l8_folder_path = args.train_l8_folder_path
    test_l8_folder_path = args.test_l8_folder_path
    val_l8_folder_path = args.val_l8_folder_path
    testval_l8_folder_path = args.testval_l8_folder_path
    lucas_csv_path = args.lucas_csv_path
    climate_csv_folder_path = args.climate_csv_folder_path



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
        testval_ds_w_id =  SNDataset(testval_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=test_transform, return_point_id=True)
        
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

        testval_ds_w_id = SNDatasetClimate(testval_l8_folder_path,
                                    lucas_csv_path,
                                    climate_csv_folder_path,
                                    l8_bands=bands, transform=test_transform, return_point_id=True)
        
        
    # COUNTING the csv files in the csv folder
    CSV_FILES = [f for f in os.listdir(climate_csv_folder_path) if f.endswith('.csv')]
    NUM_CLIMATE_FEATURES = len(CSV_FILES)
    NUM_CLIMATE_FEATURES



    from soilnet.soil_net import SoilNet, SoilNetLSTM


    cv_results = {"train_loss": [],
                "val_loss": [],
                "MAE": [],
                "RMSE": [],
                "R2": []
        }


    from datetime import date, datetime
    # Format the date and time
    now = datetime.now()
    run_name = now.strftime("D_%Y_%m_%d_T_%H_%M")
    print("Start Date and Time:", run_name)
    # create a folder called 'results' in the current directory if it doesn't exist
    if not os.path.exists('results'):
        os.mkdir('results')
        
        
    SEEDS = [1] #[1, 4, 69, 75, 79, 128, 474, 786, 2048, 3333]



    best_mae = 1000 # just a big number, since our data is normalized between 0 and 1, mae is between 0 and 1 too.
    best_seed = SEEDS[0]
    for idx, seed in enumerate(SEEDS):
        print(tc.BOLD_BAKGROUNDs.PURPLE, f"CROSS VAL {idx+1}", tc.ENDC)
        
        
        train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        val_dl = DataLoader(val_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        #model = SoilNetFC(cnn_in_channels=12, regresor_input_from_cnn=1024, hidden_size=128).to(device)
        architecture = "101+GLAM" if USE_SPATIAL_ATTENTION else "101"
        if USE_LSTM_BRANCH:
            model = SoilNetLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch= CNN_ARCHITECTURE, reg_version= REG_VERSION,
                            cnn_in_channels=len(bands), regresor_input_from_cnn=1024,
                            lstm_n_features= NUM_CLIMATE_FEATURES, lstm_n_layers= 2, lstm_out= 128,
                            hidden_size=128).to(device)
        else:
            model = SoilNet(use_glam=USE_SPATIAL_ATTENTION, cnn_arch = CNN_ARCHITECTURE, reg_version= REG_VERSION,
                        cnn_in_channels=len(bands), regresor_input_from_cnn=1024, hidden_size=128).to(device)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Saving the model on the last epoch
        save_model_path = f"results/RUN_{run_name}.pth.tar"
        
        results = train(model, train_dl, test_dl, val_dl,
                        torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
                        RMSELoss(), epochs=NUM_EPOCHS, lr_scheduler=LR_SCHEDULER,
                        save_model_path= save_model_path,
                        save_model_if_mae_lower_than= best_mae,
                        )

        
        cv_results['train_loss'].append(results['train_loss'])
        cv_results['val_loss'].append(results['val_loss'])
        cv_results['MAE'].append(results['MAE'][0])
        cv_results['RMSE'].append(results['RMSE'][0])
        cv_results['R2'].append(results['R2'][0])
        
        if results['MAE'][0] < best_mae:
            best_mae = results['MAE'][0]
            best_seed = seed
            print(tc.BOLD_BAKGROUNDs.GREEN, f"MAE improved to {best_mae}", tc.ENDC)
        
            

    train_arr = np.asarray(cv_results['train_loss'])
    val_arr = np.asarray(cv_results['val_loss'])



    plot_train_test_losses(train_arr,val_arr, title="Train/Validation Losses", x_label="Epochs", y_label="RMSE",
                        min_max_bounds= True, tight_x_lim= True,
                        train_legend = "Train", test_legend = "Validation",
                        save_path=f"results/RUN_{run_name}.png")


    import csv

    # Format the date and time
    now = datetime.now()
    finish_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Finish Date and Time:", finish_string)


    cv_results_full = {}
    cv_results_full['MAE_MEAN'] = np.mean(cv_results['MAE'])
    cv_results_full['RMSE_MEAN'] = np.mean(cv_results['RMSE'])
    cv_results_full['R2_MEAN'] = np.mean(cv_results['R2'])
    cv_results_full['MAE_MEAN'],cv_results_full['RMSE_MEAN'],cv_results_full['R2_MEAN']


    cv_results_full['USE_LSTM_BRANCH'] = USE_LSTM_BRANCH
    cv_results_full['NUM_CLIMATE_FEATURES'] = NUM_CLIMATE_FEATURES if USE_LSTM_BRANCH else None
    cv_results_full['CSV_FILES'] = CSV_FILES if USE_LSTM_BRANCH else None
    cv_results_full['NUM_WORKERS'] = NUM_WORKERS
    cv_results_full['TRAIN_BATCH_SIZE'] = TRAIN_BATCH_SIZE
    cv_results_full['TEST_BATCH_SIZE'] = TEST_BATCH_SIZE
    cv_results_full['LEARNING_RATE'] = LEARNING_RATE
    cv_results_full['NUM_EPOCHS'] = NUM_EPOCHS
    cv_results_full['LR_SCHEDULER'] = LR_SCHEDULER
    cv_results_full['CNN_ARCHITECTURE'] = CNN_ARCHITECTURE
    cv_results_full['REG_VERSION'] = REG_VERSION
    cv_results_full['USE_SPATIAL_ATTENTION'] = USE_SPATIAL_ATTENTION
    cv_results_full['Best Seed'] = best_seed
    cv_results_full['SEEDS'] = SEEDS
    cv_results_full['OC_MAX'] = OC_MAX
    cv_results_full['USE_SRTM'] = USE_SRTM
    cv_results_full['TIME'] = {"start": start_string, "finish": finish_string}
    cv_results_full['cv_results'] = cv_results


    with open(f"results/RUN_{run_name}.json", "w") as fp:
        json.dump(cv_results_full, fp, indent=4)
    
    print("Training finished and results saved!")

elif __name__ != "__main__" and __name__ != "__mp_main__":
    print("The train.py is not being run as main file. \n To run it, please use the command: \n python train.py")
    
    
# load_checkpoint(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),filename=f"results/RUN_{run_name}_{USER}.pth.tar")
# model.eval()
# print("Model loaded")



# test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
