import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path):
	if isinstance(path, str):
		df = pd.read_csv(path)
		return df
	else:
		print("Enter path in String type")

def calculate_probabilities(df_data, df_class, class_label):
	unique_column_values = df_data[class_label].unique().tolist()
	unique_column_values


def calculate_prior_probabilities(df_class):
	classes = df_class.Species.unique().tolist()
	total_instances = df_class.count()
	probabilities = {}
	for x in classes:	
	  	x_count = df_class['Species'].tolist().count(x)
	  	prob = (x_count / total_instances).tolist()
	  	probabilities[x] = prob[0]
	return probabilities
		

def calculating_posterior_probabilities(df_data, df_class):
	data_labels = df_data.columns.tolist()
	posterior_probabilities = {}
	# for x in data_labels:
	# 	posterior_probabilities[x] = calculate_probabilities(df_data, df_class, x)
	posterior_probabilities[data_labels[0]] = calculate_probabilities(df_data, df_class, data_labels[0])







def fit_data(df_data, df_labels):
	prior_probabilities = calculate_prior_probabilities(df_labels)
	calculating_posterior_probabilities(df_data, df_labels)
	
	


df_data = load_data('/home/sheharyar/Datasets/Iris/Iris-original.csv')
#X = df_data.iloc[ : , 1:len(df_data.columns)-1]
#Y = df_data.iloc[:, len(df_data.columns)-1:]
X = df_data.drop(['Id', 'Species'], axis = 1)
#Y = pd.DataFrame(df_data['Species'], columns = ['Species'])
Y = pd.DataFrame(df_data['Species'], columns = ['Species'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
# print(x_train)
# print(y_test)
fit_data(x_train, y_train)


