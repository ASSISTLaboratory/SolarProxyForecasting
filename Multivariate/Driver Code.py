"""
Created on Tue Dec 12 11:57:56 2023
author: Joshua Daniell
This script takes the original SOLFSMY SET data and prepares striped sampled data for multivariate model

Striped sampling requires week long sets to be input
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
#%% Configuration
horizon=6 # Number of days prediction is required (6-day is what SET used)
noIP = 4 # Number of inputs (4 Solar Drivers used for JB2008, F10.7, S10.7, M10.7, Y10.7)
saved_data_dir = 'Input Data/Prepped LSTM Sets/Lookback' # Path to save prepared data to
combination_type = 'stack' # Could also use average to provide mean prediction value
make_predictions=True # Variable used to control whether predictions are made or not, if not used script can still post process predictions

debugging = True # Used to verify this driver code (uses the first 100 samples only of the input)

for noTS in range(7,23,3): # noTS is length of lookback (Number Of Time-Steps)
    horizon=6 # Number of days to forecast drivers for 
    noIP = 4 # Number of drivers for input
    offset = 0
    saved_data_path = saved_data_dir+' '+str(noTS)
    
    ###
    # Load in CSV file and use the columns corresponding to the 4 variables F,S,M,Y
    index_df = pd.read_csv('Input Data/Solar_Indices.csv',index_col=0,usecols=[0,1,2,3,4]) # Drop columns associated with 81-day centered average
    index_df.index = pd.to_datetime(index_df.index) # Converting the index to datetime index
   
    
    #%%
    # Need to scale data first
    # Load in mean and STD from original training set (used to scale and unscale data)
    mean = np.array(pd.read_csv('Input Data/mean.csv',index_col=0)).T # Contains means for F10,S10,M10, and Y10
    std = np.array(pd.read_csv('Input Data/std.csv',index_col=0)).T # Contains standard deviations for F10,S10,M10, and Y10

    scaled_data = index_df.values-mean
    scaled_data = scaled_data/std
    input_sequences = []
    target_sequences = []
    
    for i in range(len(scaled_data) - noTS - horizon+1): # Truncation occurs due to noTS at beginning and end of dataset!
        input_sequences.append(scaled_data[i : i + noTS])
        target_sequences.append(scaled_data[i + noTS : i + noTS + horizon])
    

    ## MAKE SURE THAT WE UNDERSTAND THAT DATE OF FORECAST IS PREDICTING THE VALUE FOR THAT DAY AS WELL 
    # Example if forecast is made on 7/10/2023
    # Target Output 1st position is index values on 7/10/2023
    # Essentially we are making our forecast "before" any observations are made on that day!
    
    input_sequences = np.array(input_sequences) # Converting input and output to 3D for LSTM
    target_sequences = np.array(target_sequences)
    
    
    remainder =len(target_sequences) % 7
    
    batches = (len(target_sequences)//7)
    
    truncated_input = input_sequences[remainder::,:,:]
    
    truncated_output = target_sequences[remainder::,:,:]
    
    combined_sequences = np.hstack([truncated_input,truncated_output])
    

    batched_sequences = np.reshape(combined_sequences,[batches,-1,noTS+horizon,noIP])
    
    
    sets =[]
    dates =pd.date_range(start=index_df.index[0],end=index_df.index[-1],freq='7D')

    
    X_input =[]

    for i in range(batched_sequences.shape[0]):
        X_input.append(batched_sequences[i,:,:,:])

    # data save
    X_array= np.array(X_input)
    
    X = X_array[:,:,0:noTS,:]
    
    
    
    np.save(saved_data_path+' Scaled Input.npy',X)

    

    pd.DataFrame(dates).to_csv(saved_data_path+' Striped Sampling Dates.csv')


def unscale_array(prediction,mean,std):
    mean_array = np.array(mean).reshape([-1])[0]
    std_array = np.array(std).reshape([-1])[0]
    prediction=(prediction*std_array)+mean_array
    unscaled_pred=prediction
    return unscaled_pred

truncated_dates = dates[3::]
# Now that all the data is prepared, we can load models and make predictions.
if make_predictions==True:
    for noTS in range(7,23,3):
        batch_size       = 1    # Batch Size for Prediction Steps 
        architectures    = 3    # No of Top Architectures to use (Max is 3)
        iterations = 5    # Number of models to use per architecture (max is 5)
        
        folder_name = 'LSTM_Tuners'
        data_dir    = 'Input Data/Prepped LSTM Sets/Lookback '+str(noTS)+' Scaled Input.npy'
        loss_function1 = 'MSE' # 1st Loss function for set of models
        loss_function2 = 'MAE'#  2nd Loss function for set of models
        model_path_MSE  = '../Trained Models/Multivariate/Lookback '+str(noTS)+'/'+loss_function1+' Models/' # Location of trained models 
        model_path_MAE  = '../Trained Models/Multivariate/Lookback '+str(noTS)+'/'+loss_function2+' Models/'
    
        ###############################################################################
        ####################### Load LSTM Input Structures ############################
        ###############################################################################
        
        # Load LSTM Data Structures
        inputX = np.load(data_dir)
        
        print('Input Shape: ',inputX.shape)
        print('(Samples, Striped Sample Window Size, Lookback, Drivers)')
    
        
        ###############################################################################
        ############################ Define LSTM Model ################################
        ###############################################################################
        
        class LSTMonestep(tf.keras.Model):
            #### Initialize Class & Variables (REQUIRED) #########################################
            def __init__(self, model,noIP,noTS,batch_size):
                super(LSTMonestep, self).__init__()
                self.model = model
                self.noIP = noIP
                self.noTS = noTS
                self.batch_size = batch_size
                self.loss_tracker = tf.keras.metrics.Mean(name="loss") # Change these if needed
                self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
            @property
            #### Set Metrics (REQUIRED) ##########################################################
            def metrics(self):
                return [self.loss_tracker, self.val_loss_tracker]
            #### Compile Function #####################################################
            def compile(self,optimizer,loss_fn):
                super(LSTMonestep,self).compile()
                self.optimizer = optimizer
                self.loss_fn = loss_fn
            #### Call Function (REQUIRED) #############################################
            def call(self, inputs, *args, **kwargs):
                return self.model(inputs)
            ###########################################################################
            def train_step(self, data):
                # Reset State for Each Batch (Mitigate Time Gaps)
                self.model.reset_states()
                # Unpack Data (x=input & y=output/truth)
                x, y = data
                x, y = x[0], y[0]
                # Online Forecast (REQUIRED)
                for i in range(self.batch_size):
                    with tf.GradientTape() as tape:
                        Inp = tf.reshape(x[i,-self.noTS:,:],[1,self.noTS,tf.shape(x)[2]])
                        one_step = tf.reshape(self.model(Inp,training=True),[1,tf.shape(y)[1]])
                        nloss = self.loss_fn(y[i:i+1],one_step)
                    grads = tape.gradient(nloss,self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    self.loss_tracker.update_state(nloss)
                return {"loss": self.loss_tracker.result()}
            ###########################################################################
            def test_step(self, data):
                # Reset State for Each Batch (Mitigate Time Gaps)
                self.model.reset_states()
                # Unpack Data (x=input & y=output/truth)
                x, y = data
                x, y = x[0], y[0]
                # Online Forecast (REQUIRED)
                for i in range(self.batch_size):
                    with tf.GradientTape() as tape:
                        Inp = tf.reshape(x[i,-self.noTS:,:],[1,self.noTS,tf.shape(x)[2]])
                        one_step = tf.reshape(self.model(Inp,training=False),[1,tf.shape(y)[1]])
                        nloss = self.loss_fn(y[i:i+1],one_step)
                    # grads = tape.gradient(nloss,self.model.trainable_weights)  # *NOT* UPDATING WEIGHTS by Commenting Out
                    self.val_loss_tracker.update_state(nloss)
                return {"loss": self.val_loss_tracker.result()}
            
        ###############################################################################
        ################################ LSTM Prediction ##############################
        ###############################################################################
        
        #Output format has 24 columns, each set of 4 columns represents a prediction on that horizon so for example the 24th column would be Y10 prediction at a horizon of 6 days.
        
        # Columns 1:4 [F10,S10,M10,Y10]t+1, Columns 5:8 [F10,S10,M10,Y10]t+2, Columns 9:12 [F10,S10,M10,Y10]t+3, 
        # Columns 13:16 [F10,S10,M10,Y10]t+4, Columns 17:20 [F10,S10,M10,Y10]t+5, Columns 21:24 [F10,S10,M10,Y10]t+6
        if debugging==True:
            inputX = inputX[0:100,:,:,:]
            truncated_dates = truncated_dates[0:100]
        for current_architecture in range(1,architectures+1): # Have 20 models saved, using top 3 architectures
            for current_iteration in range(1,iterations+1):
                all_preds=[]
                # Load Model and compile with MSE loss
                model_name = 'architecture_'+str(current_architecture)+'_iteration_'+str(current_iteration)+'.h5'
                pred_path = 'Predictions/Lookback '+str(noTS)+'/'+'MSE_architecture_'+str(current_architecture)+'_iteration_'+str(current_iteration)+'.csv'
                            # Define model build function for tuner
                model_loaded = tf.keras.models.load_model(model_path_MSE+model_name)
                optimizer = tf.keras.optimizers.Adam()
                BS = inputX.shape[1]
                
                model = LSTMonestep(model=model_loaded,noIP=noIP,noTS=noTS,batch_size=BS)
                model.compile(optimizer=optimizer,loss_fn=tf.keras.losses.MeanSquaredError())
                for i in range(inputX.shape[0]):
                    all_preds.append(model.predict(inputX[i,:,:,:],batch_size=1,
                                        verbose=0))
                all_preds = np.array(all_preds).astype(np.float32)
                unscaled_preds = unscale_array(all_preds, mean, std)
                date_reconstructed = pd.date_range(start=truncated_dates[0],periods=len(unscaled_preds.reshape([-1,24])),freq='D')
                all_pred_df=pd.DataFrame(unscaled_preds.reshape([-1,24]),index=date_reconstructed)
                all_pred_df.to_csv('Predictions/DataFrames/Lookback '+str(noTS)+'/MSE_architecture_'+str(current_architecture)+'_iteration_'+str(current_iteration)+'.csv',header=None)
                
        for current_architecture in range(1,architectures+1): # Have 20 models saved, using top 3 architectures
            for current_iteration in range(1,iterations+1):
                all_preds=[]
                # Load Model and compile with MSE loss
                model_name = 'architecture_'+str(current_architecture)+'_iteration_'+str(current_iteration)+'.h5'
                pred_path = 'Predictions/Lookback '+str(noTS)+'/'+'MAE_architecture_'+str(current_architecture)+'_iteration_'+str(current_iteration)+'.csv'
                            # Define model build function for tuner
                model_loaded = tf.keras.models.load_model(model_path_MAE+model_name)
                optimizer = tf.keras.optimizers.Adam()
                BS = inputX.shape[1]
                
                model = LSTMonestep(model=model_loaded,noIP=noIP,noTS=noTS,batch_size=BS)
                model.compile(optimizer=optimizer,loss_fn=tf.keras.losses.MeanAbsoluteError())
                
                # Loop through all dates, no need to specify set now!
                for i in range(inputX.shape[0]):
                    all_preds.append(model.predict(inputX[i,:,:,:],batch_size=1,
                                        verbose=0))
                all_preds = np.array(all_preds).astype(np.float32)
                unscaled_preds = unscale_array(all_preds, mean, std)
                date_reconstructed = pd.date_range(start=truncated_dates[0],periods=len(unscaled_preds.reshape([-1,24])),freq='D')
                all_pred_df=pd.DataFrame(unscaled_preds.reshape([-1,24]),index=date_reconstructed)
                all_pred_df.to_csv('Predictions/DataFrames/Lookback '+str(noTS)+'/MAE_architecture_'+str(current_architecture)+'_iteration_'+str(current_iteration)+'.csv',header=None)
                
#%% Now that all predictions are made, we can now combine predictions with either averaging or stacking
#Load in all predictions and combine to produce a 180 x N x 6 array (where N is the number of samples)
# Initializing list and arrays to store predictions
lookback_date_list =[[],[],[],[],[],[]]
lookback_array_7 = np.zeros([30,all_preds.shape[0]*all_preds.shape[1],all_preds.shape[2]])
lookback_array_10 = np.zeros([30,all_preds.shape[0]*all_preds.shape[1],all_preds.shape[2]])
lookback_array_13 = np.zeros([30,all_preds.shape[0]*all_preds.shape[1],all_preds.shape[2]])
lookback_array_16 = np.zeros([30,all_preds.shape[0]*all_preds.shape[1],all_preds.shape[2]])
lookback_array_19 = np.zeros([30,all_preds.shape[0]*all_preds.shape[1],all_preds.shape[2]])
lookback_array_22 = np.zeros([30,all_preds.shape[0]*all_preds.shape[1],all_preds.shape[2]])

# 7 day Lookback work
noTS = 7
count=0
# Looping through the 7 day lookback preductions
for file in os.listdir('Predictions/DataFrames/Lookback 7/'):
    current_df = pd.read_csv('Predictions/DataFrames/Lookback 7/'+file,header=None,index_col=0)
    current_values = current_df.values.astype('float32')
    current_dates = pd.to_datetime(current_df.index) # Storing dates of predictions to ensure they are aligned
    lookback_date_list[0].append(current_dates)
    lookback_array_7[count,:,:] = current_values
    count+=1

# 10 day Lookback work
noTS = 10
count=0
# Looping through the 7 day lookback preductions
for file in os.listdir('Predictions/DataFrames/Lookback 10/'):
    current_df = pd.read_csv('Predictions/DataFrames/Lookback 10/'+file,header=None,index_col=0)
    current_values = current_df.values.astype('float32')
    current_dates = pd.to_datetime(current_df.index) # Storing dates of predictions to ensure they are aligned
    lookback_date_list[1].append(current_dates)
    lookback_array_10[count,:,:] = current_values
    count+=1

# 13 day Lookback work
noTS = 13
count=0
# Looping through the 13 day lookback preductions
for file in os.listdir('Predictions/DataFrames/Lookback 13/'):
    current_df = pd.read_csv('Predictions/DataFrames/Lookback 13/'+file,header=None,index_col=0)
    current_values = current_df.values.astype('float32')
    current_dates = pd.to_datetime(current_df.index) # Storing dates of predictions to ensure they are aligned
    lookback_date_list[2].append(current_dates)
    lookback_array_13[count,:,:] = current_values
    count+=1

# 16 day Lookback work
noTS = 16
count=0
# Looping through the 16 day lookback preductions
for file in os.listdir('Predictions/DataFrames/Lookback 16/'):
    current_df = pd.read_csv('Predictions/DataFrames/Lookback 16/'+file,header=None,index_col=0)
    current_values = current_df.values.astype('float32')
    current_dates = pd.to_datetime(current_df.index) # Storing dates of predictions to ensure they are aligned
    lookback_date_list[3].append(current_dates)
    lookback_array_16[count,:,:] = current_values
    count+=1

# 19 day Lookback work
noTS = 19
count=0
# Looping through the 19 day lookback preductions
for file in os.listdir('Predictions/DataFrames/Lookback 19/'):
    current_df = pd.read_csv('Predictions/DataFrames/Lookback 19/'+file,header=None,index_col=0)
    current_values = current_df.values.astype('float32')
    current_dates = pd.to_datetime(current_df.index) # Storing dates of predictions to ensure they are aligned
    lookback_date_list[4].append(current_dates)
    lookback_array_19[count,:,:] = current_values
    count+=1

# 22 day Lookback work
noTS = 22
count=0
# Looping through the 22 day lookback preductions
for file in os.listdir('Predictions/DataFrames/Lookback 22/'):
    current_df = pd.read_csv('Predictions/DataFrames/Lookback 22/'+file,header=None,index_col=0)
    current_values = current_df.values.astype('float32')
    current_dates = pd.to_datetime(current_df.index) # Storing dates of predictions to ensure they are aligned
    lookback_date_list[5].append(current_dates)
    lookback_array_22[count,:,:] = current_values
    count+=1


#%% Combination
# Combine arrays of size 30 to a "global" array of size 180 

prediction_set = np.vstack([lookback_array_7,lookback_array_10,lookback_array_13,lookback_array_16,lookback_array_19,lookback_array_22]) #Combining the first dimension of predictions
prediction_dates = lookback_date_list[-1][0] # Loading in dates from prediction

# This section deals with stacking the prediction, or using a set of weights created from validation data to optimally combine predictions
if combination_type=='stack':
    #Stacking method creates a weight associated for each output of each model (180x24 weights) these are used to combine predictions!
    prediction_weights =np.load('MV-MLE Stacking Weights.npy') # Set of 180x24 weights from initial work which are used as the best way to combine forecasts (stacking method)
    prediction_stacked =np.zeros([prediction_set.shape[1],prediction_set.shape[2]]) # Array which stores the stacked prediction
    for i in range(prediction_set.shape[-1]):
        current_output = prediction_set[:,:,i]
        current_weights = prediction_weights[i,:].reshape([1,-1])
        combined_output = np.dot(current_weights,current_output)
        prediction_stacked[:,i]=combined_output
    prediction_combined=prediction_stacked
elif combination_type=='average':
    prediction_combined=np.mean(prediction_set,axis=0)
    
#%% Turning the prediction array into a single dataframe and saving into the Predictions/ directory!
column_names=['F10','S10','M10','Y10',
         'F10 t+1','S10 t+1','M10 t+1','Y10 t+1',
         'F10 t+2','S10 t+2','M10 t+2','Y10 t+2',
         'F10 t+3','S10 t+3','M10 t+3','Y10 t+3',
         'F10 t+4','S10 t+4','M10 t+4','Y10 t+4',
         'F10 t+5','S10 t+5','M10 t+5','Y10 t+5',]

combined_prediction_df = pd.DataFrame(prediction_combined,index=prediction_dates,columns=column_names)

#Saving full array
np.save('Predictions/Prediction_set.npy',prediction_set.astype('float32'))

#Saving combined prediction
combined_prediction_df.to_csv('Predictions/'+combination_type+'_prediction.csv')


