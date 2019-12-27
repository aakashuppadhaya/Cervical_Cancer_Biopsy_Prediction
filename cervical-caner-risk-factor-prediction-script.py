# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def main():
    df= pd.read_csv("../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv")
    print(df)
    #handle_special_char(df)
    data_anatomy(df)
    categorical_features(df)
    numeric_features(df)
    replace_sep_char_with_null(df)
    get_null_features(df)
    type_converison(df)
    data_spliting(df)
   

def handle_special_char(df):
    print(df.shape)

def data_anatomy(df):
    print(df.shape)
    print(df.info())

def categorical_features(df):
    df_cat_features_list=list(set(df.columns)-set(df._get_numeric_data().columns))
    print(df_cat_features_list)
    cat_values_count(df_cat_features_list,df)

def numeric_features(df):
    df_num_features_list=list(df._get_numeric_data().columns)
    print(df_num_features_list)

def cat_values_count(df_cat_features_list,df):
    for i in df_cat_features_list:
        print(df[i].value_counts())

def replace_sep_char_with_null(df):
    df.replace('?', np.NaN,inplace=True)

def get_null_features(df):
    null_feature_list=df.columns[df.isna().any()].to_list()
    replace_null_with_zero(null_feature_list,df)

def replace_null_with_zero(null_feature_list,df):
    for i in null_feature_list:
        if (df[i].isnull().sum()>0):
            df[i].fillna(0,inplace=True)
    df.info()
            

def type_converison(df):
    df=df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
    df.info()
    
def data_spliting(df):
    X=df.loc[:,:'Citology']
    Y=df['Biopsy']
    print(len(X))
    print(len(Y))
    data_stratification(X,Y)
    
    
def data_stratification(X,Y):
    X_train_stratify, X_test_stratify, y_train_stratify, y_test_stratify = train_test_split(X,Y, test_size=0.2, random_state=42,stratify=Y)
    model_building(X_train_stratify,y_train_stratify,X_test_stratify,y_test_stratify)
    
def model_building(X_train_stratify,y_train_stratify,X_test_stratify,y_test_stratify):
    clf = RandomForestClassifier(n_estimators=100, random_state=5, n_jobs=-1)
    knn = KNeighborsClassifier()
    dtree = DecisionTreeClassifier()
    svm = SVC()
    clf.fit(X_train_stratify, y_train_stratify)
    knn.fit(X_train_stratify, y_train_stratify)
    dtree.fit(X_train_stratify, y_train_stratify)
    svm.fit(X_train_stratify, y_train_stratify)
    model_performance_score(clf,X_test_stratify,y_test_stratify)
    model_performance_score(knn,X_test_stratify,y_test_stratify)
    model_performance_score(dtree,X_test_stratify,y_test_stratify)
    model_performance_score(svm,X_test_stratify,y_test_stratify)
    model_saving(clf,knn,svm,dtree)
    
    

def model_performance_score(clf,X_test_stratify,y_test_stratify):
    y_pred = clf.predict(X_test_stratify)
    accuracy = clf.score(X_test_stratify,y_test_stratify)
    print(accuracy)
    ''''precision = precision_score(y_test_stratify,y_pred)
    print(precision)
    recall = recall_score(y_test_stratify,y_pred)
    print(recall)
    f1 = f1_score(y_test_stratify,y_pred)
    print(f1)'''
    
def model_saving(clf,knn,svm,dtree):
    joblib.dump(clf, 'Random_Forest.pkl')
    joblib.dump(knn, 'knn.pkl')
    joblib.dump(svm, 'SVM.pkl')
    joblib.dump(dtree, 'Decision_Tree.pkl')
    
    
if __name__=="__main__":
    main()

