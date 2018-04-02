#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tester import dump_classifier_and_data, test_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_payments', 'deferral_payments', 'exercised_stock_options', \
                 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'expenses', \
                 'other', 'director_fees', 'loan_advances', 'deferred_income', 'long_term_incentive', \
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'from_messages', \
                 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



print "\n", "######### total numbers of data point:"
print len(data_dict)

print "\n", "######### total numbers of features:"
for features in data_dict.values():
    print len(features)
    print features
    break

print "\n", "######### total number of poi:"
pois = [x for x, y in data_dict.items() if y['poi']]
print len(pois)

print ""
print "######### check the nan data:"
for key, value in data_dict.items():
    nan_num = 0
    for i in value.values():
        if i == "NaN":
            nan_num = nan_num + 1

    if nan_num > 15:
        print key, ":", nan_num

print "\n", "######### find the outliers:"
salary_list = []
bonus_list = []

for features in data_dict.values():
    # plt.scatter(features["salary"], features["bonus"])

    #
    if features["salary"] == "NaN" or features["bonus"] == "NaN":
        continue
    salary_list.append(features["salary"])
    bonus_list.append(features["bonus"])

#
bonus_list.sort()
salary_list.sort()

print "\n", "######### the top five:"
print salary_list[-5:]
print bonus_list[-5:]

print "\n", "######### the bottom five:"
print salary_list[0:5]
print bonus_list[0:5]

#
# plt.title('The original dataset:')
# plt.xlabel("salary")
# plt.ylabel("bonus")

#
print "\n", "######### top of the salary and bonus:"
print "######### show the problem data point:"
for key, value in data_dict.items():
    if value["salary"] == salary_list[-1]:
        print ""
        print key, "'s salary : ", value["salary"]

    if value["bonus"] == bonus_list[-1]:
        print ""
        print key, "'s bonus : ", value["bonus"]

    if key == "THE TRAVEL AGENCY IN THE PARK" or key == "LOCKHART EUGENE E":
        print ""
        print key, ":"
        print value

#


### Task 2: Remove outliers

### Store to my_dataset for easy export below.
my_dataset = data_dict

my_dataset.pop("TOTAL")
my_dataset.pop("THE TRAVEL AGENCY IN THE PARK")
my_dataset.pop("LOCKHART EUGENE E")
print "\n", "after(remove outliers), the total number of the dataset:"
print len(my_dataset)

#
# for features in my_dataset.values():
#	plt.scatter(features["salary"], features["bonus"])

# plt.title('After removing the outliers:')
# plt.xlabel("salary")
# plt.ylabel("bonus")
# plt.show()

#


df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace('NaN', np.nan, inplace=True)

df.info()


#
# for features in my_dataset.values():
#	colors = features["poi"]
#	if colors == True:
#		colors = "red"
#	else:
#		colors = "blue"
#	plt.scatter(features["from_poi_to_this_person"], features["from_this_person_to_poi"], c=colors,alpha=0.5)

# plt.xlabel("from_poi_to_this_person")
# plt.ylabel("from_this_person_to_poi")
# plt.show()

### Task 3: Create new feature(s)
def computeFraction(poi_messages, all_messages):
    fraction = 0.

    if poi_messages == "NaN" or all_messages == "NaN":
        return 0

    fraction = float(poi_messages) / all_messages

    return fraction


for i in data_dict:
    my_dataset[i]['fraction_from_poi'] = computeFraction(my_dataset[i]['from_poi_to_this_person'],
                                                         my_dataset[i]['to_messages'])
    my_dataset[i]['fraction_to_poi'] = computeFraction(my_dataset[i]['from_this_person_to_poi'],
                                                       my_dataset[i]['from_messages'])

# for features in my_dataset.values():
#	colors = features["poi"]
#	if colors == True:
#		colors = "red"
#	else:
#		colors = "blue"
#	plt.scatter(features["fraction_from_poi"], features["fraction_to_poi"], c=colors,alpha=0.5)

# plt.xlabel("fraction_from_poi")
# plt.ylabel("fraction_to_poi")
# plt.show()

##
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".



def Select_K_Best(data_dict, features_list, k):
    data_array = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data_array)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    tuples = zip(features_list[1:], scores)
    k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)

    return k_best_features[:k]


print "\n", Select_K_Best(my_dataset, features_list, 5)

features_list = ["poi"] + [x[0] for x in Select_K_Best(my_dataset, features_list, 5)]

print "\n", features_list


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

##


print "\n", features[0:3]

scaler = MinMaxScaler()
features_new = scaler.fit_transform(features)

print "\n", features_new[0:3]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

labels = np.asarray(labels)



features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## 1. GaussianNB


t1 = time()

clf_NB = GaussianNB()
parm = {}

clf_NB = Pipeline([('scaler', scaler), ('gnb', clf_NB)])
gs = GridSearchCV(clf_NB, parm)
gs.fit(features_train, labels_train)

clf_NB = gs.best_estimator_

print "\nGaussianNB score:\n", clf_NB.score(features_train, labels_train)
print "GaussianNB score time:", round(time() - t1, 3), "s"

##  Test Point


print "\nGaussianNB:\n", test_classifier(clf_NB, my_dataset, features_list)

## 2. Decision Tree Classifier


t2 = time()

parms = {'criterion': ['gini', 'entropy'], \
         'min_samples_split': [2, 5, 10, 20], \
         'max_depth': [None, 2, 5, 10], \
         'splitter': ['random', 'best'], \
         'max_leaf_nodes': [None, 5, 10, 20]}

clf_DT = tree.DecisionTreeClassifier()

gs = GridSearchCV(clf_DT, parms)
gs.fit(features_train, labels_train)

clf_DT = gs.best_estimator_

print "\nDecision Tree Classifier\n", clf_DT.score(features_train, labels_train)
print "Decision Tree Classifier:", round(time() - t2, 3), "s"

##  Test Point
print "\nDecision Tree Classifier Test Point:\n", test_classifier(clf_DT, my_dataset, features_list)

## 3. SVM



parms = {'svc__kernel': ('linear', 'rbf'), 'svc__C': [1.0, 2.0]}

t3 = time()
clf_SVC = SVC()

pipeline2 = Pipeline([('scaler', scaler), ('svc', clf_SVC)])
# a = pipeline.fit(features_train,labels_train)
gs = GridSearchCV(pipeline2, parms)
gs.fit(features_train, labels_train)

clf_SVC = gs.best_estimator_

print "\nSVM \n", clf_SVC.score(features_train, labels_train)
print "Decision Tree Classifier:", round(time() - t3, 3), "s"

##  Test Point
print "\nSVM:\n", test_classifier(clf_SVC, my_dataset, features_list)

### 4. RandomForest


t4 = time()

print '\nRandomForest\n'
clf_RF = RandomForestClassifier()
parameters = {'criterion': ['gini', 'entropy'], \
              'max_depth': [None, 2, 5, 10], \
              'max_leaf_nodes': [None, 5, 10, 20], \
              'n_estimators': [1, 5, 10, 50, 100]}
gs = GridSearchCV(clf_RF, parameters)
gs.fit(features_train, labels_train)

clf_RF = gs.best_estimator_

print "\nRandomForest:\n", clf_RF.score(features_train, labels_train)
print "Decision Tree Classifier:", round(time() - t4, 3), "s"

##  Test Point
print "\nRandomForest:\n", test_classifier(clf_RF, my_dataset, features_list)



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = clf_NB

dump_classifier_and_data(clf, my_dataset, features_list)
