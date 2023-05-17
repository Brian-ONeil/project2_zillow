#####imports

import numpy as np
import os
import seaborn as sns
import scipy.stats as stat
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import warnings
warnings.filterwarnings("ignore")

import wrangle as wra
import env
import explore as exp
import evaluate as ev

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import statsmodels.api as sm

from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

####functions

def plot_residuals(y, yhat):
    '''creates a residual plot'''
    
    # scatter is my actuals
    plt.scatter(train.squarefeet, train.residuals)

    # lineplot is my regression line
    plt.plot(train.squarefeet, train.yhat)

    plt.xlabel('x = squarefeet')
    plt.ylabel('y = residuals')
    plt.title('OLS linear model (n = 5000)')
    #plt.text(5000, 5000000, 'n = 5000', fontsize=12, color='red')

    plt.show()
    
def regression_errors(y, yhat):
    """
    Calculate regression error metrics for a given set of actual and predicted values.
    
    Parameters:
    y (array-like): Actual values of the response variable
    yhat (array-like): Predicted values of the response variable
    
    Returns:
    tuple: A tuple of the sum of squared errors (SSE), explained sum of squares (ESS),
           total sum of squares (TSS), mean squared error (MSE), and root mean squared error (RMSE)
    """
    # Calculate SSE, ESS, TSS, MSE, and RMSE
    SSE = np.sum((y - yhat)**2)
    ESS = np.sum((yhat - np.mean(y))**2)
    TSS = SSE + ESS
    MSE = SSE / len(y)
    RMSE = np.sqrt(MSE)
    
    # Return the results as a tuple
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    """
    Calculate regression error metrics for the baseline model that always predicts the mean of y.
    
    Parameters:
    y (array-like): Actual values of the response variable
    
    Returns:
    tuple: A tuple of the sum of squared errors (SSE), mean squared error (MSE), and root mean squared error (RMSE)
    """
    # Calculate the mean of y
    mean_y = np.mean(y)
    
    # Calculate SSE, MSE, and RMSE for the baseline model
    SSE = np.sum((y - mean_y)**2)
    MSE = SSE / len(y)
    RMSE = np.sqrt(MSE)
    
    # Return the results as a tuple
    return SSE, MSE, RMSE

def better_than_baseline(y, yhat):
    """
    Check if the sum of squared errors (SSE) for the model is less than the SSE for the baseline model.
    
    Parameters:
    y (array-like): Actual values of the response variable
    yhat (array-like): Predicted values of the response variable
    
    Returns:
    bool: True if the SSE for the model is less than the SSE for the baseline model, otherwise False
    """
    # Calculate SSE for the model and the baseline model
    SSE_model = np.sum((y - yhat)**2)
    mean_y = np.mean(y)
    SSE_baseline = np.sum((y - mean_y)**2)
    
    # Check if the SSE for the model is less than the SSE for the baseline model
    if SSE_model < SSE_baseline:
        return True
    else:
        return False

def select_kbest(X, y, k):
    """
    Select the top k features based on the SelectKBest class and return their names.
    
    Parameters:
    X (array-like): The predictors
    y (array-like): The target variable
    k (int): The number of features to select
    
    Returns:
    list: A list of the names of the top k selected features
    """
    # Create a SelectKBest object and fit it to the data
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    
    # Get the indices of the top k selected features
    idxs_selected = selector.get_support(indices=True)
    
    # Get the names of the top k selected features
    features_selected = list(X.columns[idxs_selected])
    
    # Return the names of the top k selected features
    return features_selected

def rfe(X, y, k):
    """
    Select the top k features based on the RFE class and return their names.
    
    Parameters:
    X (array-like): The predictors
    y (array-like): The target variable
    k (int): The number of features to select
    
    Returns:
    list: A list of the names of the top k selected features
    """
    # Create a linear regression model
    model = LinearRegression()
    
    # Create an RFE object and fit it to the data
    selector = RFE(model, n_features_to_select=k)
    selector.fit(X, y)
    
    # Get the indices of the top k selected features
    idxs_selected = selector.get_support(indices=True)
    
    # Get the names of the top k selected features
    features_selected = list(X.columns[idxs_selected])
    
    # Return the names of the top k selected features
    return features_selected

def metrics_reg(y, yhat):
    '''
    send in y_true, y_pred and returns rmse, r2
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

### Plot Functions

def plot_hist_subplots(df):
    """
    Creates a subplot of histograms for each column in the dataframe.
    """
    # Set figure size.
    plt.figure(figsize=(16, 6))

    # Loop through columns.
    for i, col in enumerate(df.columns):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1

        # Create subplot.
        plt.subplot(2, 4, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        plt.hist(df[col])

    # Display the plot.
    plt.show()
    
def plot_box_value_counts(df):
    """
    Creates a boxplot and value counts for each column (except the last one) in the dataframe.
    """
    # Loop through columns (except the last one).
    for col in df.columns[:-1]:
        print(col)

        # Create boxplot.
        sns.boxplot(data=df, x=col)
        plt.show()

        # Display value counts.
        print(df[col].value_counts().sort_index())
        print()
        
### Statistical Visuals and Functions

def lotsize_plot_lmplot(train):
    """
    Creates an lmplot of lotsize_sf vs. taxvalue for a sample of 5000 rows from the training set.
    """
    # Create a sample of 5000 rows from the training set.
    train_sample = train.sample(n=5000, random_state=42)

    # Create lmplot.
    sns.lmplot(x='lotsize_sf', y='taxvalue', data=train_sample)
    plt.show()
       
def lotsize_pearson_corr(train):
    """
    Calculates the Pearson correlation coefficient and p-value for lotsize_sf vs. taxvalue and prints the results.
    """
    # Calculate the Pearson correlation coefficient and p-value.
    corr, pval = stat.pearsonr(train['lotsize_sf'], train['taxvalue'])

    # Print the correlation coefficient and p-value.
    print(f"Pearson correlation coefficient: {corr:.3f}")
    print(f"P-value: {pval:.3f}")
    
def finished_sf_plot_lmplot(train):
    """
    Creates an lmplot of finished_sf vs. taxvalue for a sample of 5000 rows from the training set.
    """
    # Create a sample of 5000 rows from the training set.
    train_sample = train.sample(n=5000, random_state=42)

    # Create lmplot.
    sns.lmplot(x='finished_sf', y='taxvalue', data=train_sample)
    plt.show()
    
def finished_sf_pearson_corr(train, col1, col2):
    """
    Calculates the Pearson correlation coefficient and p-value for two columns in the dataframe and prints the results.
    """
    # Calculate the Pearson correlation coefficient and p-value.
    corr, pval = stat.pearsonr(train[col1], train[col2])

    # Print the correlation coefficient and p-value.
    print(f"Pearson correlation coefficient: {corr:.3f}")
    print(f"P-value: {pval:.3f}")
    
def bedroomcnt_plot_catplot(train):
    """
    Creates a boxplot of taxvalue vs. bedroomcnt for a sample of 5000 rows from the training set.
    """
    # Set figure size.
    plt.figure(figsize=(18, 10))

    # Create a sample of 5000 rows from the training set.
    train_sample2 = train.sample(n=5000, random_state=42)

    # Get bedroom count order.
    bedroom_order = sorted(train_sample2['bedroomcnt'].unique())

    # Create catplot with box plot.
    sns.catplot(x='bedroomcnt', y='taxvalue', data=train_sample2, kind='box', order=bedroom_order)

    # Display the plot.
    plt.show()   
    
def bedroomcnt_anova_results(train):
    """
    Converts the 'bedroomcnt' column to categorical and performs an ANOVA test on the taxvalue for each category.
    Prints the F-statistic and p-value.
    """
    # Convert the 'bedroomcnt' column to categorical
    train['bedroomcnt'] = train['bedroomcnt'].astype(str)

    # Perform the ANOVA test
    result = stat.f_oneway(train[train['bedroomcnt'] == '0']['taxvalue'], 
                             train[train['bedroomcnt'] == '1']['taxvalue'], 
                             train[train['bedroomcnt'] == '2']['taxvalue'], 
                             train[train['bedroomcnt'] == '3']['taxvalue'], 
                             train[train['bedroomcnt'] == '4']['taxvalue'], 
                             train[train['bedroomcnt'] == '5']['taxvalue'], 
                             train[train['bedroomcnt'] == '6']['taxvalue'], 
                             train[train['bedroomcnt'] == '7']['taxvalue'], 
                             train[train['bedroomcnt'] == '8']['taxvalue'], 
                             train[train['bedroomcnt'] == '9']['taxvalue'])

    # Print the results
    print("F-statistic:", result[0])
    print("P-value:", result[1])
    
def garagecnt_plot_catplot(train):
    """
    Creates a boxplot of taxvalue vs. garagecarcnt for a sample of 5000 rows from the training set.
    """
    # Set figure size.
    plt.figure(figsize=(18, 10))

    # Create a sample of 5000 rows from the training set.
    train_sample2 = train.sample(n=5000, random_state=42)

    # Get garage car count order.
    garage_order = sorted(train_sample2['garagecarcnt'].unique())

    # Create catplot with box plot.
    sns.catplot(x='garagecarcnt', y='taxvalue', data=train_sample2, kind='box', order=garage_order)

    # Display the plot.
    plt.show()  
    
def garagecnt_anova_results(train):
    """
    Performs an ANOVA test on the taxvalue for each garagecarcnt category and prints the F-statistic and p-value.
    """
    # Perform ANOVA test.
    f_statistic, p_value = stat.f_oneway(train['taxvalue'][train['garagecarcnt'] == 0],
                                          train['taxvalue'][train['garagecarcnt'] == 1],
                                          train['taxvalue'][train['garagecarcnt'] == 2],
                                          train['taxvalue'][train['garagecarcnt'] == 3],
                                          train['taxvalue'][train['garagecarcnt'] == 4],
                                          train['taxvalue'][train['garagecarcnt'] == 5])

    # Print the results.
    print('F-Statistic:', f_statistic)
    print('P-Value:', p_value) 
    
    

### Modeling Functions

def linear_regression(X_train_scaled, X_train, y_train, X_validate, y_validate, X_test, y_test, baseline):
    """
    Performs linear regression on the scaled X_train and y_train data, calculates evaluation metrics for the model,
    and returns a dataframe with the model performance.
    """
    # Make a linear regression model
    lr = LinearRegression()

    # Fit the model to the training data
    lr.fit(X_train_scaled, y_train)

    # Use the model to make predictions
    yhat = lr.predict(X_train_scaled)

    # Create an array of baseline values
    baseline_array = np.repeat(baseline, len(y_train))

    # Calculate evaluation metrics for the baseline model
    rmse, r2 = ev.metrics_reg(y_train, baseline_array)

    # Create a dataframe with the model performance
    metrics_df = pd.DataFrame(data=[
        {
            'model':'baseline',
            'rmse':rmse,
            'r2':r2
        }
    
    ])

    #OLS
    #intial ML model
    lr1 = LinearRegression()
    
    #make it
    rfe = RFE(lr1, n_features_to_select=4)
    
    #fit it
    rfe.fit(X_train, y_train)
    
    #use it on train
    X_train_rfe = rfe.transform(X_train)
    
    #use it on validate
    X_val_rfe = rfe.transform(X_validate)
        
    #fit the thing
    lr1.fit(X_train_rfe, y_train)

    #use the thing (make predictions)
    pred_lr1 = lr1.predict(X_train_rfe)
    pred_val_lr1 = lr1.predict(X_val_rfe)
    
    pred_lr1[:10]
    
    #train
    ev.metrics_reg(y_train, pred_lr1)
    
    #validate
    rmse, r2 = ev.metrics_reg(y_validate, pred_val_lr1)
    rmse, r2
    
    #add to my metrics df
    metrics_df.loc[1] = ['ols_1', rmse, r2]
    metrics_df
    
    #Multiple Regression
    
    #make it
    lr2 = LinearRegression()

    #fit it on our RFE features
    lr2.fit(X_train, y_train)

    #use it (make predictions)
    pred_lr2 = lr2.predict(X_train)

    #use it on validate
    pred_val_lr2 = lr2.predict(X_validate)
    
    #train 
    ev.metrics_reg(y_train, pred_lr2)
    
    #validate
    rmse, r2 = ev.metrics_reg(y_validate, pred_val_lr2)
    
    #add to my metrics df
    metrics_df.loc[2] = ['ols', rmse, r2]
    
    #LassoLars
    
    #make it
    lars = LassoLars(alpha=0)

    #fit it
    lars.fit(X_train, y_train)

    #use it
    pred_lars = lars.predict(X_train)
    
    pd.Series(lars.coef_, index=lars.feature_names_in_)
    
    #make it
    lars = LassoLars(alpha=1)

    #fit it
    lars.fit(X_train, y_train)

    #use it
    pred_lars = lars.predict(X_train)
    pred_val_lars = lars.predict(X_validate)
    
    #train
    ev.metrics_reg(y_train, pred_lars)
    
    #validate
    rmse, r2 = ev.metrics_reg(y_validate, pred_val_lars)
    
    #add to my metrics df
    metrics_df.loc[3] = ['lars', rmse, r2]
    
    
    #make it
    lr2 = LinearRegression()

    #fit it on our RFE features
    lr2.fit(X_train, y_train)

    #use it (make predictions)
    pred_lr2 = lr2.predict(X_train)

    #use it on validate
    pred_val_lr2 = lr2.predict(X_validate)
    
    #Polynomial Regression
    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)
    
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)
    
    #make it
    pr = LinearRegression()

    #fit it
    pr.fit(X_train_degree2, y_train)

    #use it
    pred_pr = pr.predict(X_train_degree2)
    pred_val_pr = pr.predict(X_validate_degree2)
    
    #train
    ev.metrics_reg(y_train, pred_pr)
    
    #validate
    rmse, r2 = ev.metrics_reg(y_validate, pred_val_pr)
    
    #add to my metrics df
    metrics_df.loc[4] = ['poly_2', rmse, r2]
    
    #GLM
    
    #make it
    glm = TweedieRegressor(power=1, alpha=0)
    
    #fit it
    glm.fit(X_train, y_train)
    
    #use it
    pred_glm = glm.predict(X_train)
    pred_val_glm = glm.predict(X_validate)
    
    #train
    ev.metrics_reg(y_train, pred_glm)
    
    #validate
    rmse, r2 = ev.metrics_reg(y_validate, pred_val_glm)
    
    metrics_df.loc[5] = ['glm',rmse,r2]
    
    # evaluate on best model
    
    #use it
    pred_test = pr.predict(X_test_degree2)
    pred_test
    
    rmse, r2 = ev.metrics_reg(y_test, pred_test)

    metrics_df.loc[6] = ['test',rmse,r2]
    
    return metrics_df







