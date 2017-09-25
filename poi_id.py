
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[3]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[4]:

enron_data = pd.DataFrame.from_dict(data_dict, orient = 'index')


# Data Exploration

# In[5]:

enron_data.head()


# In[6]:

print "There are a total of {} people in the dataset." .format(len(enron_data.index)) 
print "Out of which {} are POI and {} Non-POI." .format(enron_data['poi'].value_counts()[True], 
                                                 enron_data['poi'].value_counts()[False])
print "Total number of email plus financial features are {}. 'poi' column is our label." .format(len(enron_data.columns)-1)


# Enron dataset is really messy and has a lot of missing values (NaN). All the features have missing values. Some features have more than 50% of their values missing, as we can see from the frequency of NaN from the table below. NaNs are coerced to 0 for training our algorithm later.

# In[7]:

enron_data.describe().transpose()


# In[11]:

enron_data.replace(to_replace='NaN', value=0.0, inplace=True)


# In[16]:

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi']
for key in data_dict.keys():
    for value in data_dict[key]:
        if value in features_list:
            continue
        features_list.append(value)
    break

import pprint
pprint.pprint(features_list)


# In[17]:

features_list.remove('email_address')
pprint.pprint(features_list)


# In[18]:

for feature in features_list:
    cnt=0
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            cnt+=1
    print feature + " -> " + str(cnt)


# In[19]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

for feature in features_list:
    maxi = 0
    cnt=0
    for key in data_dict:
        cnt+=1
        point=data_dict[key][feature]
        if point>maxi and point != 'NaN':
            maxi = point
            name = key
        plt.scatter(point, cnt)
    plt.xlabel(feature)
    plt.show()
    print name
    print maxi
    print "\n ------------------------------------------------------------------------------------ \n"


# In[20]:

### Task 2: Remove outliners
data_dict.pop('TOTAL')


# In[21]:

for feature in features_list:
    maxi = 0
    cnt=0
    for key in data_dict:
        cnt+=1
        point=data_dict[key][feature]
        if point>maxi and point != 'NaN':
            maxi = point
            name = key
        plt.scatter(point, cnt)
    plt.xlabel(feature)
    plt.show()
    print name
    print maxi
    print "\n ------------------------------------------------------------------------------------ \n"


# In[22]:

for key in data_dict.keys():
    print key


# In[23]:

data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


# In[24]:

features_list.remove('restricted_stock_deferred')
features_list.remove('director_fees')
features_list.remove('loan_advances')


# In[25]:

for key in data_dict.keys():
    try:
        data_dict[key]['fraction_from_this_person_to_poi'] = float(data_dict[key]['from_this_person_to_poi']
                                                              )/data_dict[key]['from_messages']
    except:
        data_dict[key]['fraction_from_this_person_to_poi'] = 'NaN'
        
    try:
        data_dict[key]['fraction_from_poi_to_this_person'] = float(data_dict[key]['from_poi_to_this_person']
                                                              )/data_dict[key]['to_messages']
    except:
        data_dict[key]['fraction_from_poi_to_this_person'] = 'NaN'


# In[26]:

features_list.append('fraction_from_this_person_to_poi')
features_list.append('fraction_from_poi_to_this_person')
features_list.remove('from_this_person_to_poi')
features_list.remove('from_poi_to_this_person')
features_list.remove('from_messages')
features_list.remove('to_messages')


# In[27]:

pprint.pprint(features_list)


# In[28]:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[34]:

### Task 3: Create new feature(s)

# Bonus-salary ratio
for employee, features in data_dict.iteritems():
    if features['bonus'] == "NaN" or features['salary'] == "NaN":
        features['bonus_salary_ratio'] = "NaN"
    else:
        features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])

# from_this_person_to_poi as a percentage of from_messages
for employee, features in data_dict.iteritems():
    if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN":
        features['from_this_person_to_poi_percentage'] = "NaN"
    else:
        features['from_this_person_to_poi_percentage'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])

# from_poi_to_this_person as a percentage of to_messages
for employee, features in data_dict.iteritems():
    if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN":
        features['from_poi_to_this_person_percentage'] = "NaN"
    else:
        features['from_poi_to_this_person_percentage'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])

### Impute missing email features to mean
email_features = ['to_messages',
                   'from_poi_to_this_person',
                   'from_poi_to_this_person_percentage',
                   'from_messages',
                   'from_this_person_to_poi',
                   'from_this_person_to_poi_percentage',
                   'shared_receipt_with_poi']
# Imputing email_fearures variable
from collections import defaultdict
email_feature_sums = defaultdict(lambda:0)
email_feature_counts = defaultdict(lambda:0)

for employee, features in data_dict.iteritems():
    for ef in email_features:
        if features[ef] != "NaN":
            email_feature_sums[ef] += features[ef]
            email_feature_counts[ef] += 1

email_feature_means = {}
for ef in email_features:
    email_feature_means[ef] = float(email_feature_sums[ef]) / float(email_feature_counts[ef])

for employee, features in data_dict.iteritems():
    for ef in email_features:
        if features[ef] == "NaN":
            features[ef] = email_feature_means[ef]

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


# In[35]:

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Potential pipeline steps
scaler = MinMaxScaler()
select = SelectKBest()
dtc = DecisionTreeClassifier()
svc = SVC()
knc = KNeighborsClassifier()

# Load pipeline steps into list
steps = [
         # Preprocessing
         # ('min_max_scaler', scaler),
         
         # Feature selection
         ('feature_selection', select),
         
         # Classifier
         ('dtc', dtc)
         # ('svc', svc)
         # ('knc', knc)
         ]

# Create pipeline
pipeline = Pipeline(steps)

# Parameters to try in grid search
parameters = dict(
                  feature_selection__k=[2, 3, 5, 6], 
                  dtc__criterion=['gini', 'entropy'],
                  dtc__max_depth=[None, 1, 2, 3, 4],
                  dtc__min_samples_split=[1, 2, 3, 4, 25],
                  dtc__class_weight=[None, 'balanced'],
                  dtc__random_state=[42]
                
                  )


# In[49]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

# Create training sets and test sets
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

# Cross-validation for parameter tuning in grid search 
sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )

# Create, fit, and make predictions with grid search
gs = GridSearchCV(pipeline,
                  param_grid=parameters,
                  scoring="f1",
                  cv=sss,
                  error_score=0)
gs.fit(features_train, labels_train)
labels_predictions = gs.predict(features_test)

# Pick the classifier with the best tuned parameters
clf = gs.best_estimator_
print "\n", "Best parameters are: ", gs.best_params_, "\n"

# Print features selected and their importances
features_selected=[features_list[i+1] for i in clf.named_steps['feature_selection'].get_support(indices=True)]
scores = clf.named_steps['feature_selection'].scores_
importances = clf.named_steps['dtc'].feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'The ', len(features_selected), " features selected and their importances:"
for i in range(len(features_selected)):
    print "feature no. {}: {} ({}) ({})".format(i+1,features_selected[indices[i]],importances[indices[i]], scores[indices[i]])

# Print classification report (focus on precision and recall)
report = classification_report( labels_test, labels_predictions )
print(report)


# In[40]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy', max_depth =2, min_samples_split=2, min_samples_leaf=6)

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:



