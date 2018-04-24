#!/usr/bin/python

import sys
import pickle
sys.path.append("E:/Machine Learning/data/ud120-projects/tools")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#通过之前迷你项目的完成，初步选定以下特征作为分析的对象
features_list = ['poi','salary','bonus','from_this_person_to_poi','from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
#打开后在Python中可以看出数据集有146个数据点，每个数据点有20个特征,标签值为poi,同时可以看出很多特征的值都为NaN
with open("E:/Machine Learning/data/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
#数据集中poi的数量    
a=0
for person_name in data_dict:
    if data_dict[person_name]["poi"]==1:
        a=a+1
print a


### Task 2: Remove outliers
#查看异常值，在查看salary与bonus特征的可视化时发现了异常值，移除（total,TRAVEL AGENCY IN THE PARK,LOCKHART EUGENE E)
#同时辨认出另外的异常值，但不应该移除，值得关注，在enron61702insiderpay.pdf中确认是 SKILLING JEFFREY K和LAY KENNETH L
#这两位都有至少 5 百万美元的奖金，以及超过 1 百万美元的工资，需要重点关注而不应该移除
import matplotlib.pyplot

data_dict.pop("TOTAL",0)
data_dict.pop( "TRAVEL AGENCY IN THE PARK",0)
data_dict.pop("LOCKHART EUGENE E",0)

features = ["salary", "bonus","poi"]
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    poi = point[2]
    if poi==1:
        matplotlib.pyplot.scatter( salary, bonus, color='r' )
    if poi==0:
        matplotlib.pyplot.scatter( salary, bonus, color='b' )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Task 3: Create new feature(s)
#使用from_poi和to_poi在邮件中的占比更能说明问题,在可视化结果中可以看出，to_poi_ratio小于0.2的时候，基本上都不是poi
for employee, persons in data_dict.iteritems():
    if persons['from_this_person_to_poi'] == 'NaN' or persons['from_messages'] == 'NaN':
        persons['to_poi_ratio'] = 'NaN'
    else:
        persons['to_poi_ratio'] = float(persons['from_this_person_to_poi']) / float(persons['from_messages'])
        
    if persons['from_poi_to_this_person'] == 'NaN' or persons['to_messages'] == 'NaN':
        persons['from_poi_ratio'] = 'NaN'
    else:
        persons['from_poi_ratio'] = float(persons['from_poi_to_this_person']) / float(persons['to_messages'])

#移除了零点的异常值                
'''
for key,value in data_dict.items():
    if value['from_poi_ratio']=='NaN'and value['to_poi_ratio']=='NaN':
        data_dict.pop(key)
        
features = ["to_poi_ratio", "from_poi_ratio","poi"]
data = featureFormat(data_dict, features)
'''

for point in data:
    to_poi_ratio = point[0]
    from_poi_ratio = point[1]
    poi=point[2]
    if poi==1:
        matplotlib.pyplot.scatter( to_poi_ratio, from_poi_ratio, c='r')
    if poi==0:
        matplotlib.pyplot.scatter( to_poi_ratio,from_poi_ratio, c='b' )

matplotlib.pyplot.xlabel("to_poi_ratio")
matplotlib.pyplot.ylabel("from_poi_ratio")
matplotlib.pyplot.show()

#特征缩放
list_salary = []
for employee in data_dict:
    if data_dict[employee]['salary'] != "NaN":
        list_salary.append([float(data_dict[employee]['salary'])])
    else:
        list_salary.append([0])
list_bonus = []
for employee in data_dict:
    if data_dict[employee]['bonus'] != "NaN":
        list_bonus.append([float(data_dict[employee]['bonus'])])
    else:
        list_bonus.append([0])
list_to_poi = []
for employee in data_dict:
    if data_dict[employee]['from_this_person_to_poi'] != "NaN":
        list_to_poi.append([float(data_dict[employee]['from_this_person_to_poi'])])
    else:
        list_to_poi.append([0])
list_from_poi = []
for employee in data_dict:
    if data_dict[employee]['from_poi_to_this_person'] != "NaN":
        list_from_poi.append([float(data_dict[employee]['from_poi_to_this_person'])])
    else:
        list_from_poi.append([0])
list_to_poi_ratio = []
for employee in data_dict:
    if data_dict[employee]['to_poi_ratio'] != "NaN":
        list_to_poi_ratio.append([float(data_dict[employee]['to_poi_ratio'])])
    else:
        list_to_poi_ratio.append([0])
list_from_poi_ratio = []
for employee in data_dict:
    if data_dict[employee]['from_poi_ratio'] != "NaN":
        list_from_poi_ratio.append([float(data_dict[employee]['from_poi_ratio'])])
    else:
        list_from_poi_ratio.append([0])
        
        
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
rescaled_salary = scaler.fit_transform(list_salary)
rescaled_bonus = scaler.fit_transform(list_bonus)
rescaled_to_poi = scaler.fit_transform(list_to_poi)
rescaled_from_poi = scaler.fit_transform(list_from_poi)
rescaled_to_poi_ratio = scaler.fit_transform(list_to_poi_ratio)
rescaled_from_poi_ratio = scaler.fit_transform(list_from_poi_ratio)

i=0
for employee,features_dict in data_dict.iteritems():
    features_dict['rescaled_salary']=rescaled_salary[i]
    features_dict['rescaled_bonus'] = rescaled_bonus[i]
    features_dict['rescaled_to_poi'] = rescaled_to_poi[i]
    features_dict['rescaled_from_poi'] = rescaled_from_poi[i]
    features_dict['rescaled_to_poi_ratio'] = rescaled_to_poi_ratio[i]
    features_dict['rescaled_from_poi_ratio'] = rescaled_from_poi_ratio[i]
    i=i+1
    
### Store to my_dataset for easy export below.
#为新特征得分与
my_dataset = data_dict
my_feature_list_origin = ['poi','rescaled_salary','rescaled_bonus','rescaled_to_poi','rescaled_from_poi']
my_feature_list_new = ['poi','rescaled_salary','rescaled_bonus','rescaled_to_poi','rescaled_from_poi','rescaled_to_poi_ratio','rescaled_from_poi_ratio']

### Extract features and labels from dataset for local testing
#选择特征
data = featureFormat(my_dataset, my_feature_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import tree
from sklearn.metrics import classification_report
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print clf.feature_importances_

from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True)
features_train_new = model.transform(features_train)
features_train_new.shape

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
'''
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=6, whiten=True).fit(features_train)

print pca.explained_variance_ratio_
first_pca = pca.components_[0]
second_pca = pca.components_[1]
third_pca = pca.components_[2]
forth_pca = pca.components_[3]
print first_pca
print second_pca
print third_pca
print forth_pca

'''
my_feature_list = ['poi','rescaled_salary','rescaled_bonus','rescaled_to_poi_ratio']
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

from sklearn import model_selection
from sklearn.metrics import accuracy_score


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print accuracy_score(pred,labels_test)

from sklearn.cluster import KMeans
clf=KMeans(n_clusters=2,n_init=10)
pred=clf.fit_predict(features_test)
print accuracy_score(pred,labels_test)


from sklearn.svm import SVC
clf=SVC()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print accuracy_score(pred,labels_test)
'''
from sklearn import linear_model
clf=linear_model.Lasso()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print accuracy_score(pred,labels_test)
print clf.coef_
'''


from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print accuracy_score(pred,labels_test)
print clf.feature_importances_
print classification_report(labels_test, pred)


'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selector = SelectKBest(chi2, k=3)
selector.fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed  = selector.transform(features_test)
features_train_transformed.shape
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
#通过GridSearchCV调整了Tree的参数
from sklearn.model_selection import GridSearchCV
parameters={'criterion':('gini', 'entropy'), 
            'splitter':('best','random'),}
tree = tree.DecisionTreeClassifier()
clf = GridSearchCV(tree, parameters)
clf.fit(features_train,labels_train)
print clf.best_estimator_
pred=clf.predict(features_test)
print accuracy_score(pred,labels_test)
print classification_report(labels_test, pred)

### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)