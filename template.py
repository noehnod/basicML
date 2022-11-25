#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/noehnod/basicML

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def load_dataset(dataset_path):
	#To-Do: csv 파일 경로에서 판다스 데이터프레임으로 가져와서 return
	return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):
	#To-Do: 받은 dataframe으로 아래의 통계적인 분석을 순서대로 리턴해라
	#feature의 숫자
	#number of data for class 0
	#number of data for class 1
	#The name of the label column is “target”


	tar_series = dataset_df['target'].value_counts()
	return len(dataset_df.columns)-1, tar_series[0], tar_series[1]
def split_dataset(dataset_df, testset_size):
	#To-Do: 받은 df를 train data, test data, train label test label로 순서대로 리턴해라
	#주어진 test size로 데이터를 나누어라
	train_set, test_set = train_test_split(dataset_df, test_size=testset_size, random_state=3)

	return train_set.drop(columns='target'), test_set.drop(columns='target'), train_set[['target']], test_set[['target']]
def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: 주어진 dataset를 사용해서 의사결정트리를 train해라
	#훈련이 끝나면 주어진 test dataset을 이용해서 performances를 평가해라
	#accuracy.precision,recall을 순서대로 포함한 세개의 퍼포먼스 metrics를 리턴해라
	tree_dc = DecisionTreeClassifier()
	tree_dc.fit(x_train, y_train.values.ravel())

	return accuracy_score(y_test, tree_dc.predict(x_test)), precision_score(y_test, tree_dc.predict(x_test)), recall_score(y_test, tree_dc.predict(x_test))
def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train.values.ravel())

	return accuracy_score(rf_cls.predict(x_test), y_test), precision_score(rf_cls.predict(x_test), y_test), recall_score(rf_cls.predict(x_test), y_test)

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)
	svm_pipe.fit(x_train, y_train.values.ravel())

	return accuracy_score(y_test, svm_pipe.predict(x_test)), precision_score(y_test, svm_pipe.predict(x_test)), recall_score(y_test, svm_pipe.predict(x_test))
def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
