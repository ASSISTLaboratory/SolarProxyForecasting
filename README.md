# Solar Driver Multi Lookback Ensembles

UV-MLE and MV-MLE are neural network ensemble approaches using LSTM or a mixture of MLP and LSTM for solar driver forecasting

UV-MLE forecasting models have been specifically created for F10.7 forecasting, having been trained on striped sampled data.
Methodology for such models (using original holdout data sets) and general results can been seen in the following Journal paper...

Daniell, J. D., & Mehta, P. M. (2023). Probabilistic solar proxy forecasting with neural network ensembles. Space Weather, 21, e2023SW003675. https://doi.org/10.1029/2023SW003675

MV-MLE forecasting models have been specifically created for simultaneous forecasting of all 4 solar drivers (F10,S10,M10,Y10) and have been trained on striped sampled data.
Methodology and results for MV-MLE can be seen in the pre-print for the Journal paper...

Joshua D Daniell, Piyush M Mehta. Probabilistic Short-Term Solar Driver Forecasting with Neural Network Ensembles. ESS Open Archive . November 08, 2023. DOI: 10.22541/essoar.169945662.27742738/v1

## Contents
Includes all necessary scripts, data, and models needed to make forecasts for all four solar drivers.

Also includes necessary enviroments for loading and using multivariate and univariate models.


## Installation
Users can reference https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment for dealing with Anaconda environments

For Univariate Models (UV-MLE), users must install and use the Univariate environment, which is an Anaconda environment saved as a .yml file

For MultiVariate Models (MV-MLE), users must install and use the Multivariate environment, which is an Anaconda environment saved as a .yml file

Additional issues may arise with incompatible TensorFlow or Keras versions, users should check the [website](https://mckayward.github.io/floyd-docs/guides/environments/) for compatability between TF, Keras, and Python.

The code is normally implemented using Spyder but can of course be ran on its own via console.


## Usage
Users must alter the noaa_radio_flux_fullset.csv or Solar_Indices.csv files to predict using new data.
It should also be noted that with striped sampling, inputs must have at least a period of 7 days to be included in the forecast!

There are 4 directories in this repo:

	1. Univariate F10/ | Scripts and files necessary for for making univariate F10 predictions
		a. Input Data/ | directory containing driver data and mean/STD used for scaling
			i. Prepped LSTM Sets/ | directory used to "hold" data after it has been pre-processed. Contains .npy input arrays (4D) for striped LSTMs and contains associated dates
			ii. MEAN_STD.csv | Simple file containing mean and STD of original training set, used to scale data before predictions and unscale after preditions
			iii. noaa_radio_flux_fullset.csv | File containing original CSV driver data. If operations are needed, this is the file which needs to be altered to create "new" forecasts.
		
		b. Predictions/ | Directory used to store predictions after they are made, sorted into DataFrames for each variable lookback. Contains CSV files associated with each model's prediction ... 
				 also contains combined forecasts in .npy file format
		c. F10 Driver Code.py | Script used to prepare data, make forecasts, and save forecasts. ***PRIMARILY USED FILE***
		d. UV-MLE Stacking Weights.npy | Array of size 180 which stores the weights used to combine models in a stacking approach. Created using original validation data. *Unused if user combines models with averaging*

	2. Multivariate/ | Scripts and files necessary for for making simultaneous driver predictions
		a. Input Data/ |
			i.  Prepped LSTM Sets/ | directory used to "hold" data after it has been pre-processed. Contains .npy input arrays (4D) for striped LSTMs and contains associated dates
			ii. mean.csv | Simple CSV file holding the mean values of the 4 drivers used to scale/unscale the data through normalization
			iii. Solar_Indices.csv | CSV file holding the Solar Driver data (and 81-day central averaged values) from 1-1-1997 to 3-23-2023, this file is used to pull data for prediction and/or training. Created from SOLFSMY.txt
			iv. SOLFSMY.txt | File created by SET https://sol.spacenvironment.net/JB2008/index.php?token=e54e295b-4db3-4db4-9be6-3438501b7dc0&state=1641180456 used to train models and to produce Solar_Indices.csv
			v. std.csv | Simple CSV file holding the standard deviation values of the 4 drivers used to scale/unscale the data through normalization
		b. Predictions/ | Directory used to store predictions after they are made, sorted into DataFrames for each variable lookback. Contains CSV files associated with each model's prediction ... 
				 also contains combined forecasts in .npy file formats after stacking or averaging. Additionally, holds a 180 x N x 24 array of predictions which can be used for uncertainty analysis.
		c. Driver Code.py | Script used to prepare data, make forecasts, and save forecasts. ***PRIMARILY USED FILE***
		d. MV-MLE Stacking Weights.npy | Array of size 180 which stores the weights used to combine models in a stacking approach. Created using original validation data. *Unused if user combines models with averaging*

	3. Trained Models/ | Directory containing the various models associated with the UV-MLE and MV-MLE approaches. Models were trained using striped sampling with a validation set and saved in .h5 format. 
		a. Multivariate/ | Directory containing the models used for simultaneous (4 driver) forecasting over a 6 day horizon. Models are separated into directories based on lookback (i.e. input dimension) ...
					additionally, models are separated into MAE and MSE directories, which are the optimization losses used during training, this results in a more diverse ensemble.
		a. Univariate F10/ | Directory containing the models used for F10.7 index forecasting over a 6 day horizon. Models are separated into directories based on lookback (i.e. input dimension) ...
					additionally, models are separated into MAE and MSE directories, which are the optimization losses used during training, this results in a more diverse ensemble.
	4. Environments/ | Directory containing necessary .yml environments for using UV-MLE and MV-MLE models. Environments were created and managed using Anaconda
		a. Univariate_env.yml | Environment necessary for running the UV-MLE models. Should include all modules needed and have correct python
		b. Multivariate_env.yml| Environment necessary for running the MV-MLE models. Should include all modules needed and have correct python

## Contributing

For major changes, please open an issue first
to discuss what you would like to change or contact the original author, Joshua Daniell (joshua_daniell@outlook.com)

## License

[CC4](https://creativecommons.org/licenses/by/4.0/)
