""" toolbox containing a couple of helpers """

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

def df_rescale(df: DataFrame, scaler: None, rescale_params: bool=True) -> DataFrame:
    """ 
    Returns rescaled dataframe. if rescale_params is set to true, it will return
    a tuple of rescaled Dataframe and rescale params  
    
    Parameters
    ----------
    df: DataFrame
        input dataframe with data
    scaler: Scaler
        scaler, if None, standard scaler is used
    rescale_params: bool
        if True rescale params will be returned as dictionary

    Returns
    -------
    dataframe containing rescaled data and optionally dataframe with rescale params
    """

    logger.debug('df_rescale')
    
    if (scaler is None) or not (isinstance(scaler,StandardScaler) or isinstance(scaler,MinMaxScaler)):
        scaler = StandardScaler()
    
    
    
    

    
