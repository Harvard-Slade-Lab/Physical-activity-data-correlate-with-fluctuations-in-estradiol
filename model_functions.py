#!/usr/bin/python

import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

class XGBregressor():
    def __init__(self, params):
        self.params = params

    def predict(self, dataset):

        x_train = dataset['x_train']
        y_train = dataset['y_train']

        x_val = dataset['x_val']
        y_val = dataset['y_val']

        xgb_model = XGBRegressor(
            subsample=0.8,
            n_estimators=400,
            max_depth=3,
            eta=0.05,
            colsample_bytree=0.7,
            n_jobs=-1,
            random_state=42
        )

        
        xgb_model.fit(x_train, y_train)
        y_pred = xgb_model.predict(x_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)

        MAPE = np.round(mape,3)
        MSE = np.round(mse,3)

        return MAPE, MSE
	
def reshape_input_dim(x_data, y_data):
	# reshape model input dimensions 
	x_data_arr = np.ones((x_data.shape[0]*x_data.shape[1], x_data.shape[2]))
	y_data_arr = np.ones((x_data.shape[0]*x_data.shape[1], 1))

	count = 0
	for idx in range(x_data.shape[0]): # subject and condition
		cur_y_data = y_data[idx]
		for cur_x_data in x_data[idx,:,:]: # loop through gait cycles
			x_data_arr[count,:] = cur_x_data
			y_data_arr[count,:] = cur_y_data
			count += 1

	return x_data_arr, y_data_arr

def load_data(params, data_list, data_path, segment):
	x_all = []
	y_all = []

	for subj_cond in data_list:
		if subj_cond == '.DS_Store':
			continue
		
		# set data path
		x_path = os.path.join(data_path, segment, subj_cond, 'x.csv')
		y_path = os.path.join(data_path, segment, subj_cond, 'y.csv')

		# import data
		x_data = np.loadtxt(x_path, delimiter=',').astype(float)
		y_data = np.loadtxt(y_path, delimiter=',').astype(float)
		
		# exclude age and gender from model input (0:age, 1:gender (0 = F), 2:weight, 3:height, 4:gait_duration)
		x_data = x_data[:,2:]
		
		x_all.append(x_data)
		y_all.append(y_data)

	x_all = np.array(x_all)
	y_all = np.array(y_all)
	x_all, y_all = reshape_input_dim(x_all, y_all)
	
	return x_all, y_all

def load_dataset(params, train_list, val_list):
	data_dir = params['data_dir']
	segment = params['sensor']
	dataset = {'x_train': [], 'y_train': [], 'x_val': [], 'y_val': []}

	x_train, y_train = load_data(params, train_list, data_dir, segment)
	x_val, y_val = load_data(params, val_list, data_dir, segment)

	dataset['x_train'] = x_train
	dataset['y_train'] = y_train
	dataset['x_val'] = x_val
	dataset['y_val'] = y_val

	return dataset