import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def get():
	df_patients = pd.read_excel(open('corona_tested_individuals.xlsx', 'rb'), sheet_name='1 - tested person data') 
	rowsCount, colCount = df_patients.shape
	print(rowsCount)

	df_patients = df_patients.dropna()
	df_patients = df_patients.drop(df_patients[df_patients.corona_result == 'אחר'].index)
	df_patients['corona_result'] = np.where(df_patients['corona_result'] == 'שלילי', 0, 1)
	df_patients['gender'] = np.where(df_patients['gender'] == 'זכר', 0, 1)
	df_patients['age_60_and_above'] = np.where(df_patients['age_60_and_above'] == 'No', 0, 1)
	df_patients['test_date'] = pd.to_datetime(df_patients.test_date)
	df_patients['test_date'] = df_patients.test_date.apply(lambda x: x.toordinal())
	df_patients['abroad'] = np.where(df_patients['test_indication'] == 'Abroad', 1, 0)
	df_patients['metConfirmed'] = np.where(df_patients['test_indication'] == 'Contact with confirmed', 1, 0)

	rowsCount, colCount = df_patients.shape
	print(rowsCount)

	# Separate majority and minority classes
	df_majority = df_patients[df_patients.corona_result==0]
	df_minority = df_patients[df_patients.corona_result==1]

	# Upsample minority class
	df_minority_upsampled = resample(df_minority, 
                                 replace=True,     	 # sample with replacement
                                 n_samples=18000,    # to match majority class
                                 random_state=10000) # reproducible results
 
	# Downsample majority class in order to balance the dataset
	df_majority_downsampled = resample(df_majority, 
	                                 replace=False,    # sample without replacement
	                                 n_samples=28000,     # to match minority class
	                                 random_state=10000) # reproducible results

	# Combine minority class with downsampled majority class
	df_patients = pd.concat([df_majority_downsampled, df_minority_upsampled])
	 
	# seperate the depended param from the independed data
	df_inputs = df_patients.drop(['corona_result', 'test_indication'], axis=1)

	x = df_inputs.values 	# inputs
	y = df_patients.iloc[:, 6].values # outputs

	return [df_inputs, x, y]
