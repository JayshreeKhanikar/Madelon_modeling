from sklearn.linear_model import LogisticRegression, Lasso, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, SVR
import csv
from sklearn.metrics import roc_auc_score, make_scorer, precision_recall_curve, auc
from scipy import stats

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from tqdm import tqdm
import psycopg2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def pr_auc_score(y_test, y_prob):
   '''
   Generates the Area Under the Curve for precision and recall. It takes in y_true and y_test_probability as parameters.
   y_prob = model.predict_proba(xtest)[:,1]'''

   precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:,1])
   return auc(recall, precision, reorder=True)


def iterative_score(df, model):
    '''This function takes a df without target, converts one column of df to target and
    takes the remining df wihtout the target column as X predictors. Gets train and test samples
    It takes a model which will be used to fit and score the test set.'''
    model = model()
    scores = {}
    for col in tqdm(df.columns):
        df_fresh = df
        target = df_fresh[col]
        df_fresh = df_fresh.drop(col, axis=1)
        
        x_train, x_test, target_train, target_test = train_test_split(df_fresh,\
                                                                      target, test_size = 0.3)
        model.fit(x_train, target_train)
        score = model.score(x_test, target_test)
        scores[col] = score
    return scores


def benchmark_score(df, target, model1, model2, model3, model4):
    '''defining a function which return auc_score for various classifiers. First we split the df and target from Madelon_Josh dataset'''
    #empty dictionary to store the score for each model
    benchmark_score = {}                   
    
    xtrain, xtest, ytrain, ytest = train_test_split(df, target, test_size = 0.3)
    
    model1.fit(xtrain, ytrain)
    model2.fit(xtrain, ytrain)
    model3.fit(xtrain, ytrain)
    model4.fit(xtrain, ytrain)
    
    benchmark_score[model1] = pr_auc_score(ytest, model1.predict_proba(xtest))
    benchmark_score[model2] = pr_auc_score(ytest, model2.predict_proba(xtest))
    benchmark_score[model3] = pr_auc_score(ytest, model3.predict_proba(xtest))
    benchmark_score[model4] = pr_auc_score(ytest, model4.predict_proba(xtest))
    
    return pd.DataFrame(benchmark_score, index= [0])