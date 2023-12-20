# -*- coding: utf-8 -*-
"""
This script will load in predictions made by ensemble methods and then apply a stacking algorithm

This script will:
    1. Load all data associated with predictions
    2. Use the validation and train set and truths to fit a linear regressors (i.e. best way to combine models!)
    3. This linear regressor (weights) will be saved for loading in later! (can be used in other combination analysis code)
    4. Use the validation and train set to fit a MLP/CNN neural network (find nonlinear best mapping to truths!)
    5. This MLP/CNN model will be saved for use later for combining models!
"""

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

#%% Configuration
set_to_use = 'Test Set'

if set_to_use== 'Train Set':
    true_set = 'Train True'

elif set_to_use=='Val Set':
    true_set = 'Val True' 

elif set_to_use=='Test Set':
    true_set = 'Test True'  

to_evaluate='MV-MLE (PCA)'

#%%



def import_predictions(prediction_path_list,prediction_id_list):
    """
    This function takes the methods used for prediction and imports the dictionaries associated with each
    This function returns a dictionary associated with each method
    """
    method_dict=dict()
    for i,method in enumerate(prediction_id_list):
        with open(prediction_path_list[i],'rb') as file:
            prediction_dict = pkl.load(file)
            if i==0 or i==1:
                prediction_dict['Train Set'] = np.array(prediction_dict['Train Set'],dtype=np.float32)
                prediction_dict['Val Set'] = np.array(prediction_dict['Val Set'],dtype=np.float32)
                prediction_dict['Test Set'] = np.array(prediction_dict['Test Set'],dtype=np.float32)
        method_dict[method]=prediction_dict
        
    with open('../../MLEM/Predictions/dataset_dates.pkl','rb') as file: # Loading dates from MV_MLE Case
        date_dict = pkl.load(file)
    return method_dict,date_dict

def import_truth_dict():
    with open('../../MLEM/Predictions/dataset_truths.pkl','rb') as file: # Loading truth values from MV_MLE Case (applies to all)
        truth_dict = pkl.load(file)
    return truth_dict
#Specifiying the paths and IDs necessary for all prediction loading
pred_paths=[          '../../Transfer Learning/Predictions/dataset_predictions.pkl',
                      '../../MLE for New Drivers/Predictions/dataset_predictions.pkl',
                      '../../MLEM/Predictions/dataset_predictions.pkl',
                      '../../MLEM/Predictions/PCA/dataset_predictions.pkl']# Add PCA once it is trained!

IDs=['Transfer Learning','UV-MLE','MV-MLE','MV-MLE (PCA)'] # Add PCA once it is included!

#Calling functions to load in all predictions, dates, and truth values!
predictions,dates = import_predictions(pred_paths, IDs)
truths = import_truth_dict()

model_list = []
for i in range(predictions[to_evaluate][set_to_use].shape[0]):
    model_list.append(predictions[to_evaluate][set_to_use][i,:,:])



weight_container=[]
for i in range(24):
    stacking_dataset = np.array(model_list)[:,:,i].T # Generating a set of weights for each variable
    truth = truths[true_set][:,i]
    weights_model = LinearRegression(positive=True)
    weights_model.fit(stacking_dataset, truth)
    weights = weights_model.coef_  # These are the weights associated with each model
    weights /= np.sum(weights)
    weight_container.append(weights)
print('Weighting Done')

#%% Save model weights (weight associated with EACH output 24)
weight_array = np.array(weight_container)

np.save('Weighting/'+to_evaluate+' Stacking Weights.npy', weight_array) # Saves the weights to a file for calling later!