from math import sqrt
import pandas as pd

#--------------------------------------------------------------------

#Function is used to convert data from Dataframe type to list in list form
def arrange_data(data):
    length = data.shape[0]
    arranged = [];
    for i in range(0,length):
        row = data.iloc[i].tolist();
        arranged.append(row)
    
    return arranged;

#Fuction is used to get final labels to find out our accuracy bu comparing our scorers
def get_lables(data):
    pred = [];
    for i in data:
        pred.append(i[-1]);
    
    return pred;

#--------------------------------------------------------------------

# Reference used
#URL: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

#--------------------------------------------------------------------

#Loading our Data
file_loc = "Bank_Personal_Loan_Modelling.csv"
bank_data = pd.read_csv(file_loc);

#Data Preprocessing
bank_data.set_index("ID",inplace=True);

bank_data.drop('ZIP Code',axis=1,inplace=True)

bank_data.drop('Experience',axis=1,inplace=True)

bank_data.drop('CCAvg',axis=1,inplace=True)

final_data = arrange_data(bank_data);

from sklearn.model_selection import train_test_split;

X_train, X_test = train_test_split(final_data , test_size = 0.4);

y_test = get_lables(X_test);

#Making our predictions
predictions = []
for test in X_test:
    prediction = predict_classification(X_train, test, 5)
    predictions.append(prediction)

from sklearn.metrics import accuracy_score;

print("Using KNN classifier the accuracy is:",accuracy_score(y_test, predictions)*100);
