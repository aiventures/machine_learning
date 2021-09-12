""" toolbox containing a couple of helpers """

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

def df_stats(df: DataFrame, num_std: float = 2) -> DataFrame:
    """ 
    Returns some stats on columns of a given dataframe    
    In case nonnumerical columns are found they will not be populated 
    
    Parameters
    ----------
    df: DataFrame
        input dataframe with data
    num_std: float, optional
        number of standard deviations to calclulate boundaries

    Returns
    -------
    DataFrame 
        containing following columns
        data_type: data type	
        count: non NA count	
        na_prc: percentage of NA values	
        mean: mean/average
        min	: min value
        max: max value	
        std: standard deviation	
        lower: lower boundary (given in num of std deviations)
        upper: upper boundary (given in num of std deviations)	
        std_range_prc: range of [lower;upper] given in units of standard deviation
        in_range_prc: percentage of data points within boundary
    """

    logger.debug('df_stats')

    df3_types = df.dtypes

    # get stats
    num_elements = df.shape[0] # number of elements
    df_info = pd.DataFrame(columns=df.columns).transpose()
    
    # per columns: number of elements, percentage of NAs, mean, std deviation and percentage of outliers
    df_info["data_type"] = df.dtypes
    df_info["count"]= df.count()
    df_info["na_prc"]= 100 * df.isna().sum() / df.shape[0]
    df_info["mean"]= df.mean(axis=0)
    df_info["min"]= df.min(axis=0)
    df_info["max"]= df.max(axis=0)
    df_info["std"]= df.std(axis=0)
    df_info["lower"] = df_info.apply(lambda df_info:(df_info["mean"]-num_std*df_info["std"]),axis=1)
    df_info["upper"] = df_info.apply(lambda df_info:(df_info["mean"]+num_std*df_info["std"]),axis=1)
    df_info["std_range_prc"] = 100 * df_info.apply(lambda df_info:((df_info["upper"]-df_info["lower"])/df_info["mean"]),axis=1)
    df_info = df_info.transpose()
    in_range_dict = {}
    for col in df.columns:
        # skip non numerical columns
        if not np.issubdtype(df.dtypes[col],np.number):
            print(f"col {col} is non numeric")
            continue
        num_in_range = int((df[[col]][(df[col]<=df_info[col]["upper"])&(df[col]>df_info[col]["lower"])]).count())
        in_range_dict[col] = 100 * num_in_range / num_elements
    df_info = df_info.transpose()
    df_info["in_range_prc"] = pd.Series(in_range_dict) 
    df_info = df_info.round(2)

    return df_info

def df_preprocess(df: DataFrame,encode:bool=False):
    """
    Preprocesses Dataframe

    Parameters
    ----------
    df: DataFrame
        input dataframe with data
    encode: bool
        one hot encodes string columns

    Returns
    -------
    dataframe containing only numerical columns        

    """    
    logger.debug('df_preprocess')

    non_numeric_cols = []
    for col in df:
        if not np.issubdtype(df.dtypes[col],np.number):
            logger.debug(f'dropping columns {col}')            
            non_numeric_cols.append(col)
    df_out = df.drop(columns=non_numeric_cols)
 
    if encode:
        df_encoded = pd.get_dummies(df[non_numeric_cols],drop_first=True)
        df_out = pd.concat([df_out,df_encoded],axis=1)
    
    return df_out

def df_scaler_info(df, scaler) -> DataFrame:
    """ 
    Returns scaler parameter infos as dataframe 
    """
    scaler_dict = scaler.__dict__
    num_columns = df.shape[1]
    scaler_param_dict = {}
    for scaler_key,scaler_value in scaler_dict.items():
        if scaler_key.lower().startswith("n_"):
            continue
        if isinstance(scaler_value,np.ndarray) and scaler_value.shape[0]==num_columns:
            scaler_param_dict[scaler_key] = scaler_value
    scaler_params_df = DataFrame(scaler_param_dict,index=df.columns).transpose()
    return scaler_params_df

def df_rescale(df: DataFrame, scaler=None, rescale_params: bool=True,
               preprocess:bool = True,encode = False) -> DataFrame:
    """ 
    Returns rescaled dataframe (works on copy of original df). 
    if rescale_params is set to true, it will return
    a tuple of rescaled Dataframe and rescale params  
    
    Parameters
    ----------
    df: DataFrame
        input dataframe with data
    scaler: Scaler
        scaler, if None, standard scaler is used
    rescale_params: bool
        if True rescale params will be returned as dictionary
    preprocess: bool 
        if True, cleans dataframe from nonnumerical columns
    encode: bool
        if True, encodes nonnumerical columns

    Returns
    -------
    dataframe containing rescaled data and optionally dataframe with rescale params
    """

    logger.debug('df_rescale')
    
    if (scaler is None) or not (isinstance(scaler,StandardScaler) 
                             or isinstance(scaler,MinMaxScaler)
                             or isinstance(scaler,RobustScaler)):
        scaler = StandardScaler()

    df_out = df.copy()
    
    if preprocess:
        df_out = df_preprocess(df_out,encode=encode)    
    
    scaler.fit(df_out)
    transformed_data = scaler.transform(df_out)
    df_out = DataFrame(transformed_data,columns=df_out.columns)
    if rescale_params:
        df_scaler = df_scaler_info(df_out,scaler)
        return (df_out,df_scaler)
    else:
        return df_out






    

    

    

    

    
    

    
    

    
