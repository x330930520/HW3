import sklearn
import string
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from pathos.multiprocessing import ProcessingPool as Pool
'''
This function loads training data from file
'''
def load_data(fname):
	data = []
	with open(fname, 'r+') as f:
		for line in f:
			if not line.isspace():
				line = line.strip().split(',')
				data.append(line)
	return data
'''
This function loads test data from file because the first line of test file is not a data point
and at the end of every line there is a comma
'''
def load_test_data(fname):
	data = []
	with open(fname, 'r+') as f:
		for line in f:
			line = line.strip()
			line = line.strip(punctuation)
			if not line.isspace():
				line = line.strip().split(',')
				data.append(line)
	return data[1:]

'''
This function cleans the raw data and returns the features and labels in separated lists
'''
def preprocessing(data):
	set, label, temp = [], [], []
	#Remove blank space of features
	for i in range(len(data)):
		data[i] = list(map(str.strip, data[i]))
	#Remove data with unknown label and data with missing features
	#My approach of dealing with missing features is to remove them since both categorical and continuous features have missing values
	for i in range(len(data)):
		if data[i][-1] in ['<=50K','>50K'] and '?' not in data[i]:
			temp.append(data[i])
	#Remove education feature because education-num provides the same information
	for i in range(len(temp)):
		del temp[i][3]
	#Convert label to int and append them into a list, because there are only 2 labels, so I manually convert them to int values
	for x in temp:
		if x[-1] == '<=50K':
			label.append(0)
		if x[-1] == '>50K':
			label.append(1)
		#Append features into a list
		d = x[:-1]
		set.append(d)
	return [set, label]

'''
This function converts string values to corresponding int values
and convert all values to float
'''
def to_float(data):
	#Find distinct values of each feature
	workclass = set()
	marital_status = set()
	occupation = set()
	relationship = set()
	race = set()
	sex = set()
	native_country = set()
	for x in data[0]:
		workclass.add(x[1])
		marital_status.add(x[4])
		occupation.add(x[5])
		relationship.add(x[6])
		race.add(x[7])
		sex.add(x[8])
		native_country.add(x[12])
	lst1 = list(workclass)
	lst4 = list(marital_status)
	lst5 = list(occupation)
	lst6 = list(relationship)
	lst7 = list(race)
	lst8 = list(sex)
	lst12 = list(native_country)
	#Replace string features with int
	for x in data[0]:
		x[1] = lst1.index(x[1])
		x[4] = lst4.index(x[4])
		x[5] = lst5.index(x[5])
		x[6] = lst6.index(x[6])
		x[7] = lst7.index(x[7])
		x[8] = lst8.index(x[8])
		x[12] = lst12.index(x[12])
	#Convert all int values to float 
	for i in range(len(data[0])):
		data[0][i] = [float(x) for x in data[0][i]]
	return [data[0], data[1]]
	
'''
This function learns from input using decision tree and runs 13-fold cross-validation to select the best model
'''
def learn_decision_tree(data_set, label):
	#Create depths 
	depths = list(range(1,14))
	#Initialize the best model
	best_model = [None, 0, float("-inf")]
	#Create 13-fold
	kf = KFold(n_splits=13)
	track = []
	for (train, test), cdepth in zip(kf.split(data_set), depths):
        #Get training set
		train_set = [data_set[i] for i in train]
		train_label = [label[i] for i in train]
		#Get validation set
		valid_set = [data_set[i] for i in test]
		valid_label = [label[i] for i in test]
		#Learn the decision tree from data
		clf = tree.DecisionTreeClassifier(max_depth=cdepth)
		clf = clf.fit(train_set, train_label)
		#Get accuracy from the model
		accuraclabel = clf.score(valid_set, valid_label)
		#Compare accuracies
		track.append([cdepth, accuraclabel])
		if accuraclabel > best_model[2]:
			#Update the best model
			best_model = [clf, cdepth, accuraclabel]
	#Plot the graph
	fig = plt.figure()
	x = [x[0] for x in track]
	y = [x[1] for x in track]
	plt.xlabel('Depth')
	plt.ylabel('Accuracy')
	plt.title('Decision Tree')
	plt.plot(x,y)
	plt.savefig('decision_tree.png')
	return best_model

'''
This function learns from input using naive bayes and runs 10-fold cross-validation to select the best model
'''
def learn_naive_bayes(data_set, label):
    #Initialize the best model
	best_model = [None, float("-inf")]
    #Create 10-fold
	kf = KFold(n_splits=10)
	for train, test in kf.split(data_set):
		#Get training set
		train_set = [data_set[i] for i in train]
		train_label = [label[i] for i in train]
		#Get validation set
		valid_set = [data_set[i] for i in test]
		valid_label = [label[i] for i in test]
		#Create the naive bayes object
		clf = GaussianNB()
		#Learn naive bayes from data
		clf = clf.fit(train_set, train_label)
		#Get accuracy from the model
		accuracy = clf.score(valid_set, valid_label)
		#Compare accuracies
		if accuracy > best_model[1]:
			#Update the best model
			best_model = [clf, accuracy]
	return best_model
		
'''
This function learns from input using logistic regression and runs 10-fold cross-validation to select the best model
'''		
def learn_logistic_regression(data_set, label):
    #Initialize the best model
	best_model = [None, float("-inf")]
    #Create 10-fold
	kf = KFold(n_splits=10)
	for train, test in kf.split(data_set):
		#Get training set
		train_set = [data_set[i] for i in train]
		train_label = [label[i] for i in train]
		#Get validation set
		valid_set = [data_set[i] for i in test]
		valid_label = [label[i] for i in test]
		#Create the logistic regression object
		clf = linear_model.LogisticRegression()
		#Learn logistic regression from data
		clf = clf.fit(train_set, train_label)
		#Get accuracy from the model
		accuracy = clf.score(valid_set, valid_label)
		#Compare accuracies
		if accuracy > best_model[1]:
			#Update the best model
			best_model = [clf, accuracy]
	return best_model
	
'''
This function learns from input using nearest neighbors and runs 10-fold cross-validation to select the best model
'''		
def learn_nearest_neighbors(data_set, label):
    #Initialize the best model
	best_model = [None, 0, float("-inf")]
	#Create 10 K values from 1 to 10
	k_lst = list(range(1, 31))
    #Create 30-fold
	kf = KFold(n_splits=30)
	track = []
	for (train, test), k in zip(kf.split(data_set), k_lst):
		#Get training set
		train_set = [data_set[i] for i in train]
		train_label = [label[i] for i in train]
		#Get validation set
		valid_set = [data_set[i] for i in test]
		valid_label = [label[i] for i in test]
		#Create the nearest neighbors object
		clf = KNeighborsClassifier(n_neighbors=k)
		#Learn nearest neighbors from data
		clf = clf.fit(train_set, train_label)
		#Get accuracy from the model
		accuracy = clf.score(valid_set, valid_label)
		track.append([k, accuracy])
		#Compare accuracies
		if accuracy > best_model[2]:
			#Update the best model
			best_model = [clf, k, accuracy]
	#Plot the graph
	fig = plt.figure()
	x = [x[0] for x in track]
	y = [x[1] for x in track]
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('Nearest Neighbors')
	plt.plot(x,y)
	plt.savefig('nearest_neighbors.png')
	return best_model

'''
This function learns from input using adaboost and runs 13-fold cross-validation with different depths to select the best model
'''
def learn_adaboost(data_set, label):
	#Create depths 
	depths = list(range(1,14))
	#Create a list to keep track of parameters and performance
	track = []
	#Initialize the best model
	best_model = [None, 0, float("-inf")]
	#Create 13-fold
	kf = KFold(n_splits=13)
	for (train, test), cdepth in zip(kf.split(data_set), depths):
			#Get training set
			train_set = [data_set[i] for i in train]
			train_label = [label[i] for i in train]
			#Get validation set
			valid_set = [data_set[i] for i in test]
			valid_label = [label[i] for i in test]
			#Learn the adaboost based on decision tree from data
			clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=cdepth))
			clf = clf.fit(train_set, train_label)
			#Get accuracy from the model
			accuraclabel = clf.score(valid_set, valid_label)
			#Compare accuracies
			track.append([cdepth, accuraclabel])
			if accuraclabel > best_model[2]:
				#Update the best model
				best_model = [clf, cdepth, accuraclabel]
	#Plot the graph
	fig = plt.figure()
	x = [x[0] for x in track]
	y = [x[1] for x in track]
	plt.xlabel('Depth')
	plt.ylabel('Accuracy')
	plt.title('Adaboost')
	plt.plot(x,y)
	plt.savefig('adaboost.png')
	return best_model

	
'''
This function learns from input using random forest and runs 10-fold cross-validation with different depths and number of trees to select the best model
'''
	
def learn_random_forest(data_set, label):
    #Initialize the best model
	best_model = [None, 0, 0, float("-inf")]
    #Create 10-fold
	kf = KFold(n_splits=10)
	#Create 14 different depths and iterate through each of them
	depths = list(range(1,14))
	#Create a list to keep track of parameters and performance
	track = []
	for train, test in kf.split(data_set):
			#Create 10 different trees and iterate through each of them
			for t in list(range(1,10)):
				for cdepth in depths:
					#Get training set
					train_set = [data_set[i] for i in train]
					train_label = [label[i] for i in train]
					#Get validation set
					valid_set = [data_set[i] for i in test]
					valid_label = [label[i] for i in test]
					#Create the random forest
					clf = RandomForestClassifier(n_estimators = t, max_depth = cdepth)
					#Learn random forest from data
					clf = clf.fit(train_set, train_label)
					#Get accuracy from the model
					accuracy = clf.score(valid_set, valid_label)
					#Compare accuracies
					track.append([cdepth, t, accuracy])
					if accuracy > best_model[3]:
						#Update the best model
						best_model = [clf, cdepth, t, accuracy]
	#Plot the graph
	x = [x[0] for x in track]
	y = [x[1] for x in track]
	z = [x[2] for x in track]
	fig = plt.figure()
	plt.title('Random Forest')
	ax = plt.axes(projection='3d')
	xLabel = ax.set_xlabel('Depth')
	yLabel = ax.set_ylabel('Number of Trees')
	zLabel = ax.set_zlabel('Accuracy')
	ax.scatter(x,y,z)
	plt.savefig('randomforest.png')
	return best_model
	

'''
Building the SVM classifier 
'''	
def SVM(train_set, train_label, valid_set, valid_label, c):
	clf = SVC(C = c)
	clf = clf.fit(train_set, train_label)
	result = [clf, clf.score(valid_set, valid_label), c]
	return result
	
'''
This function learns from input using SVM and runs 10-fold cross-validation with different C values to select the best model (with multiprocessing)
'''	
def learn_SVM(data_set, label):
    #Initialize the best model
	best_model = [None, 0, float("-inf")]
    #Create 10-fold
	kf = KFold(n_splits=10)
	#Create different C values
	C = np.linspace(1.0, 100.0, num = 5)
	#Create lists to store different datasets
	ts = []
	tl = []
	vs = []
	vl = []
	for train, test in kf.split(data_set):
		#Get training set
		train_set = [data_set[i] for i in train]
		train_label = [label[i] for i in train]
		#Get validation set
		valid_set = [data_set[i] for i in test]
		valid_label = [label[i] for i in test]
		#Append them to different lists
		ts.append(train_set)
		tl.append(train_label)
		vs.append(valid_set)
		vl.append(valid_label)
	#Using parallel computing to reduce time
	res = Pool().map(SVM, ts, tl, vs, vl, C)
	accu = 0
	best_model = []
	#Compare results
	for x in res:
		if x[1] > accu:
			accu = x[1]
			best_model = x
	#Plot the graph
	fig = plt.figure()
	x = [x[2] for x in res]
	y = [x[1] for x in res]
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.title('SVM')
	plt.scatter(x,y)
	plt.savefig('SVM.png')
	return best_model		
	
	
if __name__ == "__main__":
	
	train_data = to_float(preprocessing(load_data('adult.data')))
	test_data = to_float(preprocessing(load_test_data('adult.test')))
	train_X = train_data[0]
	train_y = train_data[1]
	test_X = test_data[0]
	test_y = test_data[1]
	#Run and get the best decision tree accuracy from test data
	print ('\n\n')
	print ('*' * 20)
	best_tree = learn_decision_tree(train_X, train_y)
	tree_acurracy = best_tree[0].score(test_X, test_y)
	print ("Best decision tree accuracy is " + str(tree_acurracy))
	print ("Depth is " + str(best_tree[1]))
	print ('*' * 20)
	print ('\n\n')
	
	
	#Run and get the best logistic regression accuracy from test data
	print ('*' * 20)
	best_naive_bayes = learn_naive_bayes(train_X, train_y)
	naive_bayes_accuracy = best_naive_bayes[0].score(test_X, test_y)
	print ("Best logistic naive bayes is " + str(naive_bayes_accuracy))
	print ()
	#Get F1 score
	y_true_bayes = best_naive_bayes[0].predict(test_X)
	f1_bayes = f1_score(y_true_bayes, test_y)  
	print ("F1 score of naive bayes is " + str(f1_bayes))
	print ()
	#Get confusion matrix
	print ("Confusion Matrix: ")
	print (confusion_matrix(y_true_bayes, test_y))
	print ()
	print ('*' * 20)
	print ('\n\n')
	print ('*' * 20)
	
	
	
	#Run and get the best logistic regression accuracy from test data
	best_logistic_regression = learn_logistic_regression(train_X, train_y)
	logistic_regression_accuracy = best_logistic_regression[0].score(test_X, test_y)
	print ("Best logistic regression accuracy is " + str(logistic_regression_accuracy))
	print ()
	#Get F1 score
	y_true_log = best_logistic_regression[0].predict(test_X)
	f1_log = f1_score(y_true_log, test_y)  
	print ("F1 score of logistic regression is " + str(f1_log))
	print ()
	#Get confusion matrix
	print ("Confusion Matrix: ")
	print (confusion_matrix(y_true_log, test_y))
	print ()
	print ('*' * 20)
	print ('\n\n')
	print ('*' * 20)
	
	
	
	#Run and get the best nearest neighbors accuracy from test data
	best_nearest_neighbors = learn_nearest_neighbors(train_X, train_y)
	nearest_neighbors_accuracy = best_nearest_neighbors[0].score(test_X, test_y)
	print ("Best nearest neighbors accuracy is " + str(nearest_neighbors_accuracy))
	print ()
	print ("K is " + str(best_nearest_neighbors[1]))
	print ()
	#Get F1 score
	y_true_nb = best_nearest_neighbors[0].predict(test_X)
	f1_nb = f1_score(y_true_nb, test_y)  
	print ("F1 score of nearest neighbors is " + str(f1_nb))
	print ()
	#Get confusion matrix
	print ("Confusion Matrix: ")
	print (confusion_matrix(y_true_nb, test_y))
	print ('*' * 20)
	print ()
	
	#Run and get the best adaboost accuracy from test data
	print ('\n\n')
	print ('*' * 20)
	best_adaboost = learn_adaboost(train_X, train_y)
	adaboost_acurracy = best_adaboost[0].score(test_X, test_y)
	print ("Best adaboost accuracy is " + str(adaboost_acurracy))
	print ()
	print ("Depth is " + str(best_adaboost[1]))
	#Get F1 score
	y_true_adaboost = best_adaboost[0].predict(test_X)
	f1_adaboost = f1_score(y_true_adaboost, test_y)  
	print ()
	print ("F1 score of adaboost is " + str(f1_adaboost))
	print ()
	#Get confusion matrix
	print ("Confusion Matrix: ")
	print (confusion_matrix(y_true_adaboost, test_y))
	print ('*' * 20)
	print ()
	
	
	
	#Run and get the best random forest accuracy from test data
	print ('\n\n')
	print ('*' * 20)
	best_random_forest = learn_random_forest(train_X, train_y)
	random_forest_acurracy = best_random_forest[0].score(test_X, test_y)
	print ("Best random forest accuracy is " + str(random_forest_acurracy))
	print ()
	print ("Depth is " + str(best_random_forest[1]))
	print ()
	print ("Number of tree is " + str(best_random_forest[2]))
	print ()
	#Get F1 score
	y_true_random_forest = best_random_forest[0].predict(test_X)
	f1_random_forest = f1_score(y_true_random_forest, test_y)  
	print ("F1 score of random forest is " + str(f1_random_forest))
	print ()
	#Get confusion matrix
	print ("Confusion Matrix: ")
	print (confusion_matrix(y_true_random_forest, test_y))
	print ('*' * 20)
	print ('\n\n')
	
	
	#Run and get the best SVM accuracy from test data
	print ('\n\n')
	print ('*' * 20)
	best_SVM = learn_SVM(train_X, train_y)
	SVM_acurracy = best_SVM[0].score(test_X, test_y)
	print ("Best SVM accuracy is " + str(SVM_acurracy))
	print ()
	print ("C value is " + str(best_SVM[1]))
	print ()
	#Get F1 score
	y_true_SVM = best_SVM[0].predict(test_X)
	f1_SVM = f1_score(y_true_SVM, test_y)  
	print ("F1 score of SVM is " + str(f1_SVM))
	print ()
	#Get confusion matrix
	print ("Confusion Matrix: ")
	print (confusion_matrix(y_true_SVM, test_y))
	print ('*' * 20)
	print ('\n\n')