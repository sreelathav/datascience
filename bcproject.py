# Sree Latha Vallabhaneni           Student_Id: 15205032
#project under Strand 1: Statistical modelling/machine Learning
# Analysis of data, downloaded from kaggle website 
#https://www.kaggle.com/zcbmxvnyico/d/uciml/breast-cancer-wisconsin-data

#import libraries and required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sbn
from sklearn.metrics import classification_report,roc_curve, auc
from sklearn.model_selection import train_test_split

#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.gaussian_process.kernels import RBF

#Load data. 
data = pd.read_csv('Documents/dataProgPython/input/breastCancerW.csv')
data.head(10)
type(data)
#data cleaning, arrange or transform data variables  
data = data.drop("id",1)
data = data.drop("Unnamed: 32",1)
data.head(3)
data[[0]].head(10)

#checking for null values
data.apply(lambda x: sum(x.isnull()),axis=0) #no null values

data.describe()# distribution of mean values is different for the different characterstics
data.info()# size 569 * 30 float 1 catogorical -'diagnosis'
dir(data)

data.diagnosis.unique() # view categories
#map 'Diagnosis' string names(B/M) to binary(0/1)
cl = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(cl).astype(int)
data['diagnosis'].head()

#--------------------exploratory analysis--------------------#

print(data.columns) 
#10 subtitled _mean, 10 with _se and another 10 with _worst excluding diagnosis

# cut into chunks 1: _mean, 2: _se and 3: _worst and explore histograms etc
dcut1=data.ix[:,1:11].copy()  
dcut2=data.ix[:,11:21].copy() 
dcut3=data.ix[:,21:].copy()   

#check for each chunk variables
print(dcut1.columns) # variables with _mean in column name
print(dcut2.columns) # variables with _se in column name
print(dcut3.columns) #variables with _worst in column name

#standardise
dcut1 = (dcut1-dcut1.mean())/dcut1.std()
dcut2 = (dcut2-dcut2.mean())/dcut2.std()
dcut3 = (dcut3-dcut3.mean())/dcut3.std()
#histograms
dcut1.hist(bins=10)
dcut2.hist(bins=10)
dcut3.hist(bins=10)

#correlation in each chunk
dcut1.corr(method='pearson', min_periods=1)
dcut2.corr(method='pearson', min_periods=1)
dcut3.corr(method='pearson', min_periods=1)
#
#plot  predictor variables with _mean
#dcut1.boxplot()
plt.rcParams['figure.figsize']=(15,12)
sbn.boxplot(dcut1)
plt.show()
plt.savefig('Documents/dataProgPython/output/boxCut1.png')
plt.clf()
#log transform might bring down the difference in distribution
#log transform cut1  (_mean data) and see
cut1_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 
'area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean','fractal_dimension_mean']
log_dcut1 = dcut1.loc[:,cut1_columns]
log_dcut1[cut1_columns] = log_dcut1[cut1_columns].apply(np.log10)

#plot  log transformed predictor variables with _mean
plt.rcParams['figure.figsize']=(15,12)
sbn.boxplot(log_dcut1)
plt.show()
plt.savefig('Documents/dataProgPython/output/boxLogCut1.png')
plt.clf()
#
#plot  predictor variables with _se
plt.rcParams['figure.figsize']=(15,12)
sbn.boxplot(dcut2)
plt.show()
plt.savefig('Documents/dataProgPython/output/boxCut2.png')
plt.clf()
# log transform dcut2 variables
cut2_columns = ['radius_se', 'texture_se', 'perimeter_se', 
'area_se','smoothness_se', 'compactness_se', 'concavity_se',
       'concave points_se', 'symmetry_se','fractal_dimension_se']
log_dcut2=dcut2.loc[:,cut2_columns]
log_dcut2[cut2_columns] = log_dcut2[cut2_columns].apply(np.log10)


#plot  log transformed predictor variables with _se (dcut2)
plt.rcParams['figure.figsize']=(15,12)
sbn.boxplot(log_dcut2)
plt.show()
plt.savefig('Documents/dataProgPython/output/boxLogCut2.png')
plt.clf()
#
#plot  predictor variables with _worst (dcut3)
plt.rcParams['figure.figsize']=(15,12)
sbn.boxplot(dcut3)
plt.show()
plt.savefig('Documents/dataProgPython/output/boxCut3.png')

# log transform dcut3 variables
cut3_columns = [ 'radius_worst','texture_worst', 'perimeter_worst', 
'area_worst','smoothness_worst','compactness_worst', 'concavity_worst',
       'concave points_worst' , 'symmetry_worst','fractal_dimension_worst']
log_dcut3 =dcut3.loc[:,cut3_columns]
log_dcut3[cut3_columns] = log_dcut3[cut3_columns].apply(np.log10)

#plot  log transformed predictor variables (with _worst) (dcut3)
plt.clf()
plt.rcParams['figure.figsize']=(15,12)
sbn.boxplot(log_dcut3)
plt.show()
plt.savefig('Documents/dataProgPython/output/boxLogCut3.png')


####Using grid search to tune for optimal parameters for the classifiers
def fit_model(model_type,model_name,tuning_parameters):    
    clf = GridSearchCV(model_type, tuning_parameters, cv=15,
                           scoring = 'accuracy')
    clf.fit(X_train, y_train)            
    print(" Best parameters with train data for the classifier, %s" %model_name )
    print( clf.best_params_)
    print("Grid scores for train data: ")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("      %0.5f (+/-%0.03f) for %r" % (mean, std * 2, params))
    y_pred = clf.predict(X_test)
    print("     Classification report:")
    print(classification_report(y_test, y_pred))       
    return 

#prepare data for model analysis
data.head()
y = data['diagnosis']
y.head()
#Gridsearch with normal data (without droping any columns)
X = data.ix[:,1:].copy()
#Normalize the data 
X = (X-X.mean())/X.std()
X.head()
#Split dataset into Train and Test
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.25)
#-------------------fit models---------------
#1) Logistic Regression:
fit_model(LogisticRegression(),'Logit',[{'C': [10, 100, 1000,2000, 3000]}])

#2) Random Forest Classifier:
fit_model(RandomForestClassifier(n_jobs=-1),'RF',[{'n_estimators':[10,50,100,150]}])

#3) Extra Trees Classifier:
fit_model(ExtraTreesClassifier(n_jobs=-1),'ET',[{'n_estimators':[10,50,100]}])

#4) Gradient Boosting Classifier: (Optional paramaters are learning_rate=1.0,max_depth=1,random_state=0
fit_model(GradientBoostingClassifier(),'GradientBoost',[{'n_estimators':[10,50,100,150]}])

#5) Adaboosting Classifier: Optional paramaters are learning_rate=1.0,random_state=0
fit_model(AdaBoostClassifier(),'AdaBoost',[{'n_estimators':[200,300,400,450]}])
 

#Gridsearch with log transformed data (dropped columns with Concavity)
log_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 
               'area_mean','smoothness_mean', 'compactness_mean', 
               'symmetry_mean','fractal_dimension_mean',
               'radius_se', 'texture_se', 'perimeter_se', 
               'area_se','smoothness_se', 'compactness_se', 
               'symmetry_se','fractal_dimension_se',
               'radius_worst','texture_worst', 'perimeter_worst', 
               'area_worst','smoothness_worst','compactness_worst', 
               'symmetry_worst','fractal_dimension_worst']
X = data.ix[:,log_columns].apply(np.log10) #log tansforming data
X.head()

#Split dataset into Train and Test
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.25)

#-------------------fit models---------------
#1) Logistic Regression:
fit_model(LogisticRegression(),'Logit',[{'C': [10, 100, 1000,2000, 3000]}])

#2) Random Forest Classifier:
fit_model(RandomForestClassifier(n_jobs=-1),'RF',[{'n_estimators':[10,50,100,150]}])

#3) Extra Trees Classifier:
fit_model(ExtraTreesClassifier(n_jobs=-1),'ET',[{'n_estimators':[10,50,100]}])

#4) Gradient Boosting Classifier: (Optional paramaters are learning_rate=1.0,max_depth=1,random_state=0
fit_model(GradientBoostingClassifier(),'GradientBoost',[{'n_estimators':[10,50,100,150]}])

#5) Adaboosting Classifier: Optional paramaters are learning_rate=1.0,random_state=0
fit_model(AdaBoostClassifier(),'AdaBoost',[{'n_estimators':[200,300,400,450]}])


#-------------------KNN and SVM for ROC plots------------------#
X = data.ix[:,1:]
#Normalize the data 
X = (X-X.mean())/X.std()
X.head()
#Split dataset into Train and Test
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.25)

#  KNN classification using normalized data
knn3 = KNeighborsClassifier(n_neighbors=3)
#fit
knn3fit = knn3.fit(X_train, y_train)
#predict
knn3testPred = knn3fit.predict(X_test)
#get probabilities
knn3testProb = knn3fit.predict_proba(X_test)
#auc
roc_knn = roc_curve(y_test, knn3testProb[:,1]) # Returns fpr, tpr, cutoffs
knn_auc3 = auc(roc_knn[0],roc_knn[1])
#        svm fit and prediction
svm_clf = SVC(probability=True)
svm_clf_fit = svm_clf.fit(X_train, y_train)
svm_test_prob = svm_clf_fit.predict_proba(X_test)
roc_svm = roc_curve(y_test, svm_test_prob[:,1])
svm_auc = auc(roc_svm[0],roc_svm[1])

print [knn_auc3, svm_auc]
##############
plt.clf()
plt.figure()
plt.title('ROC curves for KNN, SVM classifications of Wisconcin Breast Cancer data')

ax1= plt.subplot(121)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('KNN classification')

ax1.plot(roc_knn[0],roc_knn[1],label=' (AUC = {0:0.2f})'.format(knn_auc3))        
ax1.legend(loc='lower right')

ax2= plt.subplot(122)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('SVM classification')

ax2.plot(roc_svm[0],roc_svm[1],label=' (AUC = {0:0.2f})'.format(svm_auc))         
ax2.legend(loc='lower right')
plt.show()
plt.savefig('Documents/dataProgPython/output/rocKS.png')
#========================================================