""" module to analyze bank acoount data (csv data). searches for keywords in booking texts 
    and categorizes expenses/income. Output will be a time frame with fixed periods
    that can be used to display stacked bar charts
    Imported Data and Graphics Output is categorized by a data descriptor 
    Sources:
    Color Maps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    Pandas Date Offset Alias: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    Pandas Date Range https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
    Pandas Data Frame https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    Plot Bar https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html
    Label Ticks https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
    https://stackoverflow.com/questions/63549321/matplotlib-formatting-x-axis-shows-jan-1-1970 (Date formatter not working)
 """

from dateutil.parser import parse
from dateutil.parser import ParserError
from dateutil.tz import gettz
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
import matplotlib
from sklearn.feature_extraction.text import CountVectorizer
import traceback 

# Column for date
COL_YEAR = "JJJJ"
COL_YEARMONTH = "JJJJMM"
COL_YEARWEEK = "JJJJWW"

# date interval string
TIME_BUCKET = {"WEEK":"W-MON","MONTH":"MS"}
TIME_PERIOD_COL = {"WEEK":"JJJJWW","MONTH":"JJJJMM","YEAR":"JJJJ"}
TIME_PERIOD_FMT = {"WEEK":"%Y%U","MONTH":"%Y%m","YEAR":"%Y"}

# constants and master data
THOUSAND_SEP = '.'
DECIMAL_SEP = ','
DATA_SEP = ";"
CMAP_PALETTE = "viridis"

# vocabulary per category
vocabulary = ["Store1","Store2","Store3"]
# category dict 
category_dict = {"CAT" : vocabulary}

# template data descriptor replace values
DATA_DESCRIPTOR = {"FILE_NAME":r"C:\<PATH_TO>\SampleAccount.csv",
                   "DATE_COLUMNS":["Buchungsdatum"], # Columns Containing Dates
                   "N_HEADER_LINES":3,                              # Number of header Lines in CSV
                   "N_SKIP_ROWS":6,                                 # Skip number of lines in CSV
                   "DATE_COL_ORIGINAL":"Buchungsdatum",             # use this date column as index column
                   "DATE_COL":"Datum",                              # indecx column title 
                   "DEBIT_COL":"Soll (EUR)",                        # column containing debit
                   "CREDIT_COL":"Haben (EUR)",                      # column containing credit
                   "BALANCE_COL":"Betrag_EUR",                      # column containing debit/vredit = balance
                   "TEXT_COL":"Umsatzinformation",                  # column containing booking information
                   "TIME_BUCKET":"MONTH",                           # time bucket: "MONTH" or "WEEK"
                   "DROP_COLS":['UCI','Mandat ID','Abweichender Debitor', # columns to be dropped
                                'Abweichender Kreditor','Referenznummer','Wertstellung'],
                   "DRAW_CATEGORIES":False, # draw categories or vocabulary
                   "CATEGORY_DICT":category_dict, # category-vocabulary dictionary
                   "FONTSIZE":18,                 # fontsize for printing texts
                   "TITLE":"AUSGABEN",            # title in graph
                   "X_TITLE":"Periode",           # x axis title 
                   "Y_TITLE":"Ausgabe (EUR)",     # y axis title 
                   "FIGSIZE":(15,7),              # graphics size
                   "THOUSAND_SEP":THOUSAND_SEP,   # Thousands SEPARATOR
                   "DECIMAL_SEP":DECIMAL_SEP,     # Decimals separator
                   "DATA_SEP":DATA_SEP,           # CSV data separator
                   "CMAP_PALETTE":CMAP_PALETTE    # CMAP palette name
                  }


def get_balance(debit,credit):
    """ add debit and credit columns """
    balance = 0
    if not(np.isnan(debit)):
        balance += -1. * debit
    if not(np.isnan(credit)):
        balance += credit
    return balance

def categorize_info_col(df,info_column,vocabulary):
    ''' categorize info colum of a data frame with given list of key words 
        adds result as one hot encoding style columns
        returns modified dataframe
    '''
    
    # convert all vocabiuulary items to lower case
    vocabulary_lower = list(map(str.lower, vocabulary))
    columns_map_dict = dict(zip(vocabulary_lower,vocabulary))
    
    # delete any existing target columns
    try:
        df.drop(labels=vocabulary, axis=1, inplace=True)
    except:
        pass
    
    vectorizer = CountVectorizer(lowercase=True,vocabulary=vocabulary_lower,ngram_range=(1,2))
    X = vectorizer.fit_transform(df[info_column])
    df_category =  pd.DataFrame(data=X.toarray(),columns=vectorizer.get_feature_names())
    df_category.index = df.index
    df = pd.concat([df, df_category], axis=1)
    
    # rename lowercase columns to original upper case 
    df = df.rename(columns=columns_map_dict)
    
    return df

def get_cmap_names():
    """ returns list of colormaps """
    cmap_list = []
    cmap_keys = cm.__dict__.keys()    
    for k in cmap_keys:
        if isinstance(cm.__dict__[k], matplotlib.colors.ListedColormap):
            cmap_list.append(k)
    
    return sorted(cmap_list)

def get_cmap_hexvalues(num_colors=5,palette_name="viridis",plot_palette=False):
    """ Returns list of color hex values for given palette """
    c_map = cm.get_cmap(palette_name,num_colors)
    palette = sns.color_palette(palette_name,num_colors)
    color_list = []
    for i in range(c_map.N):
        rgba = c_map(i)
        hexvalue = matplotlib.colors.rgb2hex(rgba)
        color_list.append(hexvalue)    
    if plot_palette:
        sns.palplot(color_list);        
    return color_list

def fmt_date_ticks(axis_label,fmt_string="%Y-%m-%d"):
    """ formats a tick mark according to given format string """
    label_fmt = datetime.strptime(axis_label.get_text(),'%Y-%m-%d %H:%M:%S').strftime(fmt_string)
    axis_label.set_text(label_fmt)
    return axis_label

def read_header_data(data_descriptor:dict):
    """ read leading header infos """
    df_header = pd.read_csv(data_descriptor["FILE_NAME"],sep=data_descriptor["DATA_SEP"],
                            decimal=data_descriptor["DECIMAL_SEP"],
                            delimiter=data_descriptor["DATA_SEP"],
                            nrows=data_descriptor["N_HEADER_LINES"],
                            thousands=data_descriptor["THOUSAND_SEP"])
    return df_header

def get_df_from_csv(data_descriptor:dict):
    """ reads an expense csv and transforms it to data frame """
    try:
        df = pd.read_csv(data_descriptor["FILE_NAME"],sep=data_descriptor["DATA_SEP"],
                                decimal=data_descriptor["DECIMAL_SEP"],
                                parse_dates=data_descriptor["DATE_COLUMNS"],
                                delimiter=data_descriptor["DATA_SEP"],
                                nrows=999999,skiprows=data_descriptor["N_SKIP_ROWS"],
                                thousands=data_descriptor["THOUSAND_SEP"])
    except Exception as error:
        print(f'Error reading CSV file {data_descriptor["FILE_NAME"]} : ' + repr(error))
        return
    
    
    # process master data valid for all rows
    master_data = data_descriptor.get("MASTER_DATA",None)
    if master_data is not None:
        for k,v in master_data.items():
            df[k] = v
            
    # rename master date column
    date_col = data_descriptor["DATE_COL"]
    df.rename(columns={data_descriptor["DATE_COL_ORIGINAL"]:date_col},inplace=True)
    
    # convert dates
    df[COL_YEAR]   = df[date_col].map(lambda d: d.strftime('%Y'))
    df[COL_YEARMONTH] = df[date_col].map(lambda d: d.strftime('%Y%m'))
    df[COL_YEARWEEK] = df[date_col].map(lambda d: d.strftime('%Y%U'))    
    
    # calculate balance
    debit_col = data_descriptor["DEBIT_COL"]
    credit_col = data_descriptor["CREDIT_COL"]
    balance_col = data_descriptor["BALANCE_COL"]
    balance_cols = [debit_col,credit_col]
    df[balance_col] = df[balance_cols].apply(lambda f:get_balance(f[debit_col],f[credit_col]),axis=1)    
    
    # drop valid columns
    df_cols = list(df.columns)    
    drop_cols = data_descriptor["DROP_COLS"]
    drop_cols.extend(balance_cols)
    drop_cols = list(filter(lambda c:c in drop_cols,df_cols))
    df.drop(labels=drop_cols, axis=1, inplace=True)    
    

    # set date column as index
    df.set_index(date_col,inplace=True)
    
    # create vocabulary
    category_dict = data_descriptor["CATEGORY_DICT"]
    vocabulary = []

    for v in category_dict.values():
        vocabulary.extend(v)
    
    # process sub categories
    df = categorize_info_col(df,data_descriptor["TEXT_COL"],vocabulary)
    
    # process categories
    for cat,cat_cols in category_dict.items():
        df[cat] = df[category_dict[cat]].max(axis=1)
        # write balance to category column
        df.loc[df[cat]>0,cat]=-df.loc[df[cat]>0,balance_col]

    # write total balance amount into category columns
    df_voc_balance = -df[vocabulary].multiply(df[balance_col], axis=0)
    df[vocabulary] = df_voc_balance 
    
    return df

def get_df_by_period(df:pd.DataFrame,data_descriptor:dict):
    """ Build periodical buckets for display """
    df_period = None
    time_bucket = data_descriptor.get("TIME_BUCKET","MONTH")
    time_period = TIME_BUCKET[time_bucket]
    
    df_period = df.groupby(pd.Grouper(freq=time_period)).sum()
    df_period[TIME_PERIOD_COL["YEAR"]] = df_period.index.map(lambda x: x.strftime(TIME_PERIOD_FMT["YEAR"]))
    df_period[TIME_PERIOD_COL[time_bucket]] = df_period.index.map(lambda x: x.strftime(TIME_PERIOD_FMT[time_bucket]))

    return df_period

def draw_bars(df:pd.DataFrame,data_descriptor:dict):
    """ Draw Time Period Buckets """
    
    # get drawing parameters                   
    time_bucket = data_descriptor.get("TIME_BUCKET","MONTH")
    fontsize = data_descriptor.get("FONTSIZE",18)
    title = data_descriptor.get("TITLE","")
    x_title = data_descriptor.get("X_TITLE","Abscissa")
    y_title = data_descriptor.get("Y_TITLE","Ordinate")    
    figsize = data_descriptor.get("FIGSIZE",(15,7))   
    stacked = data_descriptor.get("STACKED_BARS",True)  
    palette = data_descriptor.get("CMAP_PALETTE",CMAP_PALETTE)
    plot_palette = data_descriptor.get("PLOT_PALETTE",False)
    category_dict = data_descriptor.get("CATEGORY_DICT",{})
    
    # extract relevant columns for drawing
    vocabulary = []
    for v in category_dict.values():
        vocabulary.extend(v)        
    df_vocabulary = df[vocabulary]
    df_categories = df[category_dict.keys()]
    
    # draw categories or vocabulary
    if data_descriptor.get("DRAW_CATEGORIES",False):
        df_draw = df_categories
    else:
        df_draw = df_vocabulary
    
    # set color map
    n_cols = len(df_draw.columns)
    cmap_values = get_cmap_hexvalues(num_colors=n_cols,palette_name=palette,plot_palette=plot_palette)    
    
    ax = df_draw.plot.bar(stacked=stacked,figsize=figsize,color=cmap_values)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)    
    
    plt.legend(bbox_to_anchor=(1.15, 1));
    plt.title(title,fontsize=fontsize);
    ax.set_xlabel(x_title,fontsize=fontsize);
    ax.set_ylabel(y_title,fontsize=fontsize);    
    
    # get current ticks, adjust date formatting
    locs,labels = plt.xticks() 
    fmt_string=TIME_PERIOD_FMT[time_bucket]
    labels_fmt = [fmt_date_ticks(l,fmt_string=fmt_string) for l in labels]
    plt.xticks(locs, labels_fmt, rotation = 50);   
    
    return ax 


