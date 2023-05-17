#######IMPORTS

import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#######FUNCTIONS

zillow_query = """
        select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt,
        taxamount, fips
        from properties_2017
        where propertylandusetypeid = '261';
        """

def new_zillow_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a db_url to mySQL
    - return a df of the given query from the telco_db
    """
    url = env.get_db_url('zillow')
    
    return pd.read_sql(SQL_query, url)

def get_zillow_data(SQL_query, filename = 'zillow.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv: defaulted to telco.csv
    - outputs iris df
    """
    
    if os.path.exists(filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_zillow_data(SQL_query)

        df.to_csv(filename)
        return df

def wrangle_zillow(df):
    
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    df.rename(columns={'calculatedfinishedsquarefeet': 'squarefeet', 'taxvaluedollarcnt': 'taxvalue',
                       'fips': 'county'}, inplace=True)
    
    df.dropna(inplace=True)
    
    df[['bedroomcnt', 'squarefeet', 'taxvalue', 'yearbuilt', 'county']] = df[['bedroomcnt', 'squarefeet',
                                                                              'taxvalue', 'yearbuilt',
                                                                              'county']].astype(int)

    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    
    df = df [df.squarefeet < 25_000]
    
    df = df [df.taxvalue < df.taxvalue.quantile(.95)].copy()
    
    df = df[df.taxvalue > df.taxvalue.quantile(.001)].copy()
    
    return df


def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    train, test = train_test_split(df,
                                   test_size=.2,
                                   random_state=123,
                                   )
    train, validate = train_test_split(train,
                                       test_size=.25,
                                       random_state=123,
                                       )
    return train, validate, test

def get_X_train_val_test(train,validate, test, x_target, y_target):
    '''
    geting the X's and y's and returns them
    '''
    X_train = train.drop(columns = x_target)
    X_validate = validate.drop(columns = x_target)
    X_test = test.drop(columns = x_target)
    y_train = train[y_target]
    y_validate = validate[y_target]
    y_test = test[y_target]
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def scaler_robust(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the RobustScaler on it
    '''
    scaler = RobustScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def scaled_data_to_dataframe(X_train, X_validate, X_test):
    '''
    This function scales the data and returns it as a pandas dataframe
    '''
    X_train_columns = X_train.columns
    X_validate_columns = X_validate.columns
    X_test_columns = X_test.columns
    X_train_numbers, X_validade_numbers, X_test_numbers = scaler_robust(X_train, X_validate, X_test)
    X_train_scaled = pd.DataFrame(columns = X_train_columns)
    for i in range(int(X_train_numbers.shape[0])):
        X_train_scaled.loc[len(X_train_scaled.index)] = X_train_numbers[i]
    X_validate_scaled = pd.DataFrame(columns = X_validate_columns)
    for i in range(int(X_validade_numbers.shape[0])):
        X_validate_scaled.loc[len(X_validate_scaled.index)] = X_validade_numbers[i]
    X_test_scaled = pd.DataFrame(columns = X_test_columns)
    for i in range(int(X_test_numbers.shape[0])):
        X_test_scaled.loc[len(X_test_scaled.index)] = X_test_numbers[i]
    return X_train_scaled, X_validate_scaled, X_test_scaled










    