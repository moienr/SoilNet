from train_utils import *
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
import warnings
import config
from soilnet.soil_net import SoilNet, SoilNetLSTM, SoilNetSimCLRwRegHead, SoilNetJustLSTM
import csv
import train_utils
from train_utils import *
from datetime import date, datetime
import argparse
# Format the date and time
# create a folder called 'results' in the current directory if it doesn't exist
if not os.path.exists('results'):
	os.mkdir('results')
 
# Format the date and time
now = datetime.now()
start_string = now.strftime("%Y-%m-%d %H:%M:%S")
# print("Current Date and Time:", start_string)
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

train_l8_folder_path = config.train_l8_folder_path
test_l8_folder_path = config.test_l8_folder_path
val_l8_folder_path = config.val_l8_folder_path
lucas_csv_path = config.lucas_csv_path
climate_csv_folder_path = config.climate_csv_folder_path
SIMCLR_PATH = config.SIMCLR_PATH

EXP_NAME = 'LUCAS_Transformer_NoImage'
DATASET = 'LUCAS' # 'LUCAS', 'RaCA'
NUM_WORKERS = 2
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
LR_SCHEDULER = "step" # step, plateau or None
USE_SRTM = False
USE_SPATIAL_ATTENTION = False
CNN_ARCHITECTURE = "ViT" # vgg16 or resnet101 or "ViT" or resnet50
RNN_ARCHITECTURE = 'Transformer' # LSTM, GRU, RNN, Transformer
REG_VERSION = 1
SEEDS = [1,] # Seeds for cross-validation and reproducibility
USE_LSTM_BRANCH = False
LOG_LOSS = False
SAVE_TRAIN_DATA_METRICS = False
LOAD_SIMCLR_MODEL = False
JUST_LSTM = False # Using Only Climate Data

def parse_arguments():
	parser = argparse.ArgumentParser(description='SoilNet Training')
	parser.add_argument('-e', '--exp_name', type=str, default=EXP_NAME, help='Experiment name - helps to identify the experiment')
	parser.add_argument('-d', '--dataset', type=str, default=DATASET, choices=['LUCAS', 'RaCA'], help='Dataset name to use')
	parser.add_argument('-w', '--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')
	parser.add_argument('-trb', '--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help='Batch size for training')
	parser.add_argument('-tsb', '--test_batch_size', type=int, default=TEST_BATCH_SIZE, help='Batch size for testing')
	parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
	parser.add_argument('-ne', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
	parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER, choices=['step', 'plateau', 'None'], help='Learning rate scheduler')
	parser.add_argument('-srtm', '--use_srtm', action='store_true', default=USE_SRTM, help='Use SRTM data')
	parser.add_argument('-sa', '--use_spatial_attention', action='store_true', default=USE_SPATIAL_ATTENTION, help='Use spatial attention')
	parser.add_argument('-cnn', '--cnn_architecture', type=str, default=CNN_ARCHITECTURE, choices=['vgg16', 'resnet101', 'ViT', 'resnet50'], help='CNN architecture')
	parser.add_argument('-rnn', '--rnn_architecture', type=str, default=RNN_ARCHITECTURE, choices=['LSTM', 'GRU', 'RNN', 'Transformer'], help='RNN architecture')
	parser.add_argument('-rv', '--reg_version', type=int, default=REG_VERSION, help='Regression version')
	parser.add_argument('-seed', '--seeds', nargs='+', type=int, default=SEEDS, help='Seeds for cross-validation. input example: 1 2 3 4 5')
	parser.add_argument('-lstm', '--use_lstm_branch', action='store_true', default=USE_LSTM_BRANCH, help='Use Cliamte data - I know! the name is misleading')
	parser.add_argument('-log', '--log_loss', action='store_true', default=LOG_LOSS, help='Use logarithmic loss')
	parser.add_argument('-stm', '--save_train_data_metrics', action='store_true', default=SAVE_TRAIN_DATA_METRICS, help='Save training data metrics')
	parser.add_argument('-simclr', '--load_simclr_model', action='store_true', default=LOAD_SIMCLR_MODEL, help='Load Self-supervised model to fine-tune')
	parser.add_argument('-jlstm', '--just_lstm', action='store_true', default=JUST_LSTM, help='Use only climate data')

	args = parser.parse_args()
	return args





if __name__ == '__main__':
    
	args = parse_arguments()
	EXP_NAME = args.exp_name
	DATASET = args.dataset
	NUM_WORKERS = args.num_workers
	TRAIN_BATCH_SIZE = args.train_batch_size
	TEST_BATCH_SIZE = args.test_batch_size
	LEARNING_RATE = args.learning_rate
	NUM_EPOCHS = args.num_epochs
	LR_SCHEDULER = args.lr_scheduler
	USE_SRTM = args.use_srtm
	USE_SPATIAL_ATTENTION = args.use_spatial_attention
	CNN_ARCHITECTURE = args.cnn_architecture
	RNN_ARCHITECTURE = args.rnn_architecture
	REG_VERSION = args.reg_version
	SEEDS = args.seeds
	USE_LSTM_BRANCH = args.use_lstm_branch
	LOG_LOSS = args.log_loss
	SAVE_TRAIN_DATA_METRICS = args.save_train_data_metrics
	LOAD_SIMCLR_MODEL = args.load_simclr_model
	JUST_LSTM = args.just_lstm


	if DATASET == 'LUCAS':
		from dataset.dataset_loader import SNDataset,SNDatasetClimate, myNormalize, myToTensor, Augmentations
		OC_MAX = 87
		# OC_MAX = 560.2
	if DATASET == 'RaCA':
		from dataset.dataset_loader_us import SNDataset,SNDatasetClimate, myNormalize, myToTensor, Augmentations


		OC_MAX = 4115
  

	if JUST_LSTM:
		USE_LSTM_BRANCH = True
		USE_SPATIAL_ATTENTION = False

	if LOAD_SIMCLR_MODEL:
		if USE_LSTM_BRANCH == False:
			raise Exception("LOAD_SIMCLR_MODEL is enabled but LSTM branch is disabled. Please enable LSTM branch.")
		if JUST_LSTM:
			raise Exception("LOAD_SIMCLR_MODEL is enabled but JUST_LSTM is enabled. Please disable JUST_LSTM.")

	if LOAD_SIMCLR_MODEL:
		print("\033[91m\033[1m\033[5mWARNING!\033[0m")
		print("\033[93m Loading SimCLR Model is enabled.\
			\n This will overwrite Chosen Architectures\
			\n Also, make sure that LSTM is enabled. \033[0m")    


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
				"val_loss": [],
				"MAE": [],
				"RMSE": [],
				"R2": [],
				"train_MAE": [],
					"train_RMSE": [],
					"train_R2": []
		}



	now = datetime.now()
	run_name = now.strftime("D_%Y_%m_%d_T_%H_%M")
	print("Current Date and Time:", run_name)
	# create a folder called 'results' in the current directory if it doesn't exist
	if not os.path.exists('results'):
		os.mkdir('results')
		
		
		
	best_mae = 1000 # just a big number, since our data is normalized between 0 and 1, mae is between 0 and 1 too.
	worst_mae = 0 # just a small number, since our data is normalized between 0 and 1, mae is between 0 and 1 too.

	best_rmse = 1000 # just a big number, since our data is normalized between 0 and 1, rmse is between 0 and 1 too.
	worst_rmse = 0 # just a small number, since our data is normalized between 0 and 1, rmse is between 0 and 1 too.

	best_seed = SEEDS[0]
	worst_seed = SEEDS[0]

	for idx, seed in enumerate(SEEDS):
		print(tc.BOLD_BAKGROUNDs.PURPLE, f"CROSS VAL {idx+1}", tc.ENDC)
		
		
		train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
		test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
		val_dl = DataLoader(val_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
		
		#model = SoilNetFC(cnn_in_channels=12, regresor_input_from_cnn=1024, hidden_size=128).to(device)
		architecture = "101+GLAM" if USE_SPATIAL_ATTENTION else "101"
		if not JUST_LSTM:
			if USE_LSTM_BRANCH:
				model = SoilNetLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch= CNN_ARCHITECTURE, reg_version= REG_VERSION,
							cnn_in_channels=len(bands), regresor_input_from_cnn=1024,
							lstm_n_features= NUM_CLIMATE_FEATURES, lstm_n_layers= 2, lstm_out= 128,
							hidden_size=128, rnn_arch=RNN_ARCHITECTURE,seq_len=SEQ_LEN).to(device)
			else:
				model = SoilNet(use_glam=USE_SPATIAL_ATTENTION, cnn_arch = CNN_ARCHITECTURE, reg_version= REG_VERSION,
						cnn_in_channels=len(bands), regresor_input_from_cnn=1024, hidden_size=128).to(device)
				
		else:
			model = SoilNetJustLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch= CNN_ARCHITECTURE, reg_version= REG_VERSION,
								cnn_in_channels=len(bands), regresor_input_from_cnn=1024,
								lstm_n_features= NUM_CLIMATE_FEATURES, lstm_n_layers= 2, lstm_out= 128,
								hidden_size=128, rnn_arch=RNN_ARCHITECTURE,seq_len=SEQ_LEN).to(device)
		
		if LOAD_SIMCLR_MODEL:
			model = torch.load(SIMCLR_PATH).to(device)
			model = SoilNetSimCLRwRegHead(model, hidden_size=128, reg_version=REG_VERSION).to(device)
			
		
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		
		# Saving the model on the last epoch
		# save_model_path = f"results/RUN_{EXP_NAME}_{run_name}_{USER}_best.pth.tar"
		save_model_path = None

		
		loss_instance = RMSLELoss() if LOG_LOSS else RMSELoss()
			
		optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		results = train(model, train_dl, test_dl, val_dl,
						optimizer,
						loss_instance, epochs=NUM_EPOCHS, lr_scheduler=LR_SCHEDULER,
						save_model_path= save_model_path,
						save_model_if_mae_lower_than= best_rmse,
						save_train_data_metrics=SAVE_TRAIN_DATA_METRICS
						)

		
		cv_results['train_loss'].append(results['train_loss'])
		cv_results['val_loss'].append(results['val_loss'])
		cv_results['MAE'].append(results['MAE'][0])
		cv_results['RMSE'].append(results['RMSE'][0])
		cv_results['R2'].append(results['R2'][0])
		
		if SAVE_TRAIN_DATA_METRICS:
			cv_results['train_MAE'].append(results['train_MAE'])
			cv_results['train_RMSE'].append(results['train_RMSE'])
			cv_results['train_R2'].append(results['train_R2'])
		
		# Stop the training loop via RMSE 
		if results['RMSE'][0] < best_rmse:
			best_rmse = results['RMSE'][0]
			best_seed = seed
			print(tc.BOLD_BAKGROUNDs.GREEN, f"Best RMSE improved to {best_rmse}", tc.ENDC)
			# Save the best model
			best_model_path = f"results/RUN_{EXP_NAME}_{run_name}_best.pth.tar"
			save_checkpoint(model, optimizer=optimizer, filename=best_model_path)
			
		if results['RMSE'][0] > worst_rmse:
			worst_rmse = results['RMSE'][0]
			worst_seed = seed
			print(tc.BOLD_BAKGROUNDs.RED, f"Worst RMSE worsened to {worst_rmse}", tc.ENDC)
			# Save the worst model
			worst_model_path = f"results/RUN_{EXP_NAME}_{run_name}_worst.pth.tar"
			save_checkpoint(model, optimizer=optimizer, filename=worst_model_path)
			
		print(f"This Runs RMSE: {results['RMSE'][0]}")
		
		
		
	train_arr = np.asarray(cv_results['train_loss'])
	val_arr = np.asarray(cv_results['val_loss'])


	y_label = "RMSLE" if LOG_LOSS else "RMSE"
	plot_train_test_losses(train_arr,val_arr, title="Train/Validation Losses", x_label="Epochs", y_label=y_label,
						min_max_bounds= True, tight_x_lim= True,
						train_legend = "Train", test_legend = "Validation",
						save_path=f"results/RUN_{EXP_NAME}_{run_name}.png")

	# Format the date and time
	now = datetime.now()
	finish_string = now.strftime("%Y-%m-%d %H:%M:%S")
	print("Current Date and Time:", finish_string)


	cv_results_full = {}
	cv_results_full['MAE_MEAN'] = np.mean(cv_results['MAE'])
	cv_results_full['RMSE_MEAN'] = np.mean(cv_results['RMSE'])
	cv_results_full['R2_MEAN'] = np.mean(cv_results['R2'])
	cv_results_full['LOAD_SIMCLR_MODEL'] = LOAD_SIMCLR_MODEL
	cv_results_full['JUST_LSTM'] = JUST_LSTM
	cv_results_full['USE_LSTM_BRANCH'] = USE_LSTM_BRANCH
	# cv_results_full['USE_PRIM_CLIM'] = USE_PRIM_CLIM
	# cv_results_full['USE_SEC_CLIM'] = USE_SEC_CLIM
	cv_results_full['LOG_LOSS'] = LOG_LOSS
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


	with open(f"results/Metrics_{EXP_NAME}_{run_name}.json", "w") as fp:
		json.dump(cv_results, fp, indent=4)
		
	# Load the best model
	load_checkpoint(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),filename=f"results/RUN_{EXP_NAME}_{run_name}_best.pth.tar")

	model.eval()
	print("Best Model loaded")

	test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, csv_file=f"results/RUN_{EXP_NAME}_{run_name}_best.csv")
	print(f"Best model saved to results/RUN_{EXP_NAME}_{run_name}_best.csv")

	# Load the worst model
	load_checkpoint(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),filename=f"results/RUN_{EXP_NAME}_{run_name}_worst.pth.tar")
	model.eval()
	print("Worst Model loaded")
	test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, csv_file=f"results/RUN_{EXP_NAME}_{run_name}_worst.csv")
	print(f"Worst Model saved to results/RUN_{EXP_NAME}_{run_name}_worst.csv")


	df = pd.read_csv(f"results/RUN_{EXP_NAME}_{run_name}_best.csv")


	y_true = df['y_real']* OC_MAX
	y_pred = df['y_pred']* OC_MAX

	rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

	best_dict = {}
	best_dict['RMSE'] = rmse
	best_dict['R2'] = r2
	best_dict['RPIQ'] = rpiq
	best_dict['MAE'] = mae
	best_dict['MEC'] = mec
	best_dict['CCC'] = ccc

	# print(best_dict)
	cv_results_full['best_dict'] = best_dict


	df = pd.read_csv(f"results/RUN_{EXP_NAME}_{run_name}_worst.csv")
	# df = pd.read_csv("C:\\Users\\nkakhani\\_Multimodal\\SoilNet-7\\SoilNet-PreRelease\\results\RUN_D_2024_01_29_T_18_55_Nafiseh_worst.csv")


	y_true = df['y_real']* OC_MAX
	y_pred = df['y_pred']* OC_MAX

	rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

	worst_dict = {}
	worst_dict['RMSE'] = rmse
	worst_dict['R2'] = r2
	worst_dict['RPIQ'] = rpiq
	worst_dict['MAE'] = mae
	worst_dict['MEC'] = mec
	worst_dict['CCC'] = ccc

	# print(worst_dict)
	cv_results_full['worst_dict'] = worst_dict

	with open(f"results/RUN_{EXP_NAME}_{run_name}.json", "w") as fp:
		json.dump(cv_results_full, fp, indent=4)