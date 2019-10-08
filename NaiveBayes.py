############################### Assignment # 1 ##################################
#################### Group members: 
################################## Sheharyar Ahmad	FA16-BCS-053 ################
################################## M. Shakir Khan	FA16-BCS-037 ################
################################## Adil Shahzad		FA16-BCS-063 ################
#################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

########### FUNCTION TO LOAD DATASET ##########
def load_data(path):
	if isinstance(path, str): # checks is the argument is string or not
		df = pd.read_csv(path) # read the file and stores it in dataframe
		return df # returns the dataframe
	else:
		print("Enter path in String") #prints error

########## FUNCTION TO CALCULATE PRIOR PROBABILITIES ##########
def calculate_prior_probabilities(df_class):
	classes = df_class.Class.unique().tolist() # creates a list of unique values of class attribute
	total_instances = df_class.count() # counts total no of class attribute values
	probabilities = {} # dictionary to stroe attribute values probabilities
	for x in classes:	#loop on class value list
	  	x_count = df_class['Class'].tolist().count(x) # counts(x) counts each class value in class attribute
	  	prob = (x_count / float(total_instances)) # calculate probability of each class value
	  	probabilities[x] = prob # Stores probability in dictionary
	df_prior_probabilities = pd.DataFrame(probabilities, index = [1]) # creates dataframe using above dictionary
	df_prior_probabilities.to_csv('prior_probabilitiesOV.csv') # stores datafame in csv file

########## FUNCTION TO CALCULATE POSTERIOR PROBABLITITES OF ONE ATTRIBUTE ###########
def calculate_probabilities(df_data, df_class, class_label):
	unique_column_values = df_data[class_label].unique().tolist() # finds unique attribute values and stores it in a list
	unique_class_labels = df_class['Class'].unique().tolist() # finds unique class attribute values and stores it in a list
	probability = {} # creates a dictionary
	for value in unique_column_values: # loops through the attribute values 
		for label in unique_class_labels: # loops through the class attribute values
			counter = 0 # counts occurences of values
		 	class_label_count = df_class.Class.tolist().count(label) # class value count in class attribute
	 		for i in range(0, df_data[class_label].count()): # loop from 1st index to last index of column
	 			if  (df_data[class_label].iloc[i] == value) and (df_class['Class'].iloc[i] is label): # matches value in attribute and class in class attribute
	 		 		counter +=  1 # increments
	 		prob = counter / float(class_label_count) # calculates probability
	 		if prob == 0.0: # laplace correction
	 			prob = 1.0
	 		probability[str(value) + ' | ' + str(label)] = prob # stores it in the dictionary with key as (attribute: value | class: label )
	return probability 

########### FUNCTION TO CALCULATE POSTERIOR PROBABILITIES OF ALL ATTRIBUTES #############
def calculating_posterior_probabilities(df_data, df_class):
	data_labels = df_data.columns.tolist() # stores attribute name in a list
	posterior_probabilities = {} # dict for posterior probability of all attributes
	for x in data_labels: # loops over the attribute names list
		posterior_probabilities[x] = calculate_probabilities(df_data, df_class, x) # calls calculate function every time with new attribute name
	DataFrame = pd.DataFrame(posterior_probabilities) # creates a dataframe from above dictionary
	DataFrame['Probability_Index'] = DataFrame.index # creates a new column in dataframe and assigns it values of index
	DataFrame = DataFrame.reset_index(drop=True) # Drops the index
	DataFrame.set_index('Probability_Index', inplace=True) # sets Probability_Index attribute as new index 
	DataFrame.to_csv('posterior_probabilitiesofOV.csv') # saves dataframe in csv file

############# FUNCTION TO CALCULATE PRIOR & POSTERIOR PROBABILITIES ############
def fit_data(df_data, df_labels):
	calculate_prior_probabilities(df_labels)
	calculating_posterior_probabilities(df_data, df_labels)

############ FUNCTION TO PREDICT CLASS LABELS OF TEST DATA ############
def prediction(row_data, index_value):
	df_prior_probabilities = load_data('prior_probabilitiesOV.csv') # Loads the saved file with prior probabilities
	df_posterior = load_data('posterior_probabilitiesofOV.csv') # loads the saved file with posterior probabilities
	prior_p_columns = df_prior_probabilities.columns.tolist() # stores the column names in a list
	prior_p = {} # creates a dict
	prior_p[prior_p_columns[1]] = df_prior_probabilities[prior_p_columns[1]][0] # stores the prior probability of first class in dict with key as the name of first class
	prior_p[prior_p_columns[2]] = df_prior_probabilities[prior_p_columns[2]][0] # stores the prior probability of second class in dict with key as the name of second class	
	recurrence_events = 1 # variable to store value after mulitplying probabilities given first class
	no_recurrence_events = 1 # vairable to store value after multiplying probabilities given second class
	for column_name, value in row_data.items():  # Loops over dictionary row_data containing values from one row in y_train dataframe
		index1 = np.where(df_posterior['Probability_Index'] == str(value) + ' | ' + 'recurrence-events')[0][0] # gathers index of where value of Probability_Index is equal to our desired value
		index2 = np.where(df_posterior['Probability_Index'] == str(value) + ' | ' + 'no-recurrence-events')[0][0] # gathers index of where value of Probability_Index is equal to our desired value
		recurrence_events *= df_posterior[column_name].iloc[index1] # using the index we found in previous statement finds probability and multiplies it with variable
		no_recurrence_events *= df_posterior[column_name].iloc[index2] #using the index we found in previous statment finds probability and mulitplies it with variable
	recurrence_events_likelyhood = prior_p['recurrence-events'] * recurrence_events # finds likelyhood of first class
	no_recurrence_events_likelyhood = prior_p['recurrence-events'] * no_recurrence_events # finds likelyhood of second class
	if recurrence_events_likelyhood > no_recurrence_events_likelyhood: # condition for checking which likelyhood is greater for assinging class value
		return 'recurrence-events' 
	if no_recurrence_events_likelyhood > recurrence_events_likelyhood:
		return 'no-recurrence-events'

############# FUNCTION THAT CALL PREDICITON FUNCTION WITH EACH ROW PASSED AS AN ARGUMENT #############
def predict(df_data):
	index_values = df_data.index.tolist() # stores ytrain index into list used for getting row data
	column_names = df_data.columns.tolist() # stores ytrain columns into a list
	predictions = [] # list for storing predictions
	for i in range(0, len(index_values)): # loop over index values
		row_data = df_data.loc[[index_values[i]]].values.tolist() # retrives row data and stores it in a variable
		row_data = dict(zip(column_names, row_data[0])) # makes a dictionary with column_names as key and row data values 
		row_data_lable = prediction(row_data, index_values[i]) # calls prediction function and predicts class label
		predictions.append(row_data_lable) # stores label in a list
	pd.DataFrame(predictions, index=index_values, columns=['Class']).to_csv('predictionsOV.csv') # creates a dataframe and saves it in a csv file

############# FUNCTION THAT CALCULATES ACCURACY BY COMPARING ACTUAL lABELS AND PREDICTED LABELS ##############
def find_accuracy(df_class):
	df_predictions = load_data('predictionsOV.csv') # loads csv of predictions 
	class_predictions_list = df_predictions['Class'].tolist() # saves the predictions in a list
	class_original_list = df_class['Class'].tolist() # save original class labels in a list
	correct = 0 # counter for correct predictions
	for i in range(0, len(class_original_list)): # loop over list
		if class_original_list[i] == class_predictions_list[i]: # matches items in both list if true than increments counter
			correct += 1
	return correct / float(len(class_original_list)) # calculates accuracy and returns



#####################################################################################
#################################### Main ###########################################
#####################################################################################


df_data = load_data('breast-cancer.csv') # loads original dataset
df_data = df_data.dropna(how="any") # drop any row having null values
X = df_data.drop(['Class'], axis = 1) # removes class attribute from dataframe and stores it in X
Y = pd.DataFrame(df_data['Class'], columns = ['Class']) # Creates class attribute dataframe and stores it in Y
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=92) # splits dataset in train and test sets x_train for attributes y_train for class labels same for test

fit_data(x_train, y_train)
predict(x_test)
print('Accuracy: ' + str(find_accuracy(y_test)))










# Using SkLearn Library for naive bayes classification

# gnb = GaussianNB()
# gnb.fit(x_train, y_train.values.ravel())
# y_pred = gnb.predict(x_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



