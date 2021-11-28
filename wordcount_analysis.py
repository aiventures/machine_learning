""" module to get word count statistics """

import requests
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import traceback

# References
# Maybe it would have been easier to condense information on df basis
# https://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby

def read_file(f:str)->list:
    """ reading UTF8 txt File """
    lines = []
    try:
        with open(f,encoding="utf-8") as fp:    
            for line in fp:
                lines.append(line)
    except:
        print(f"Exception reading file {f}")
        print(traceback.format_exc())
    return lines

def save_file(fn:str,t:str)->None:
    """ saving UTF8 txt File """
    with open(fn, 'w', encoding="utf-8") as f:
        try:
            f.write(t)
        except:
            print(f"Exception writing file {fn}")
            print(traceback.format_exc())     

def get_keywords(lines:list,stop_words:list=None)->pd.Series:
    """ returns keywords and count as Series object """
    v = CountVectorizer(ngram_range=(1,1),stop_words=stop_words)
    X = v.fit_transform(lines)
    X.toarray().shape
    # get keywords 
    keyword_count = X.toarray().sum(axis=0)
    keyword_dict = dict(list(zip(v.get_feature_names(),keyword_count)))
    keywords = pd.Series(data=keyword_dict)
    keywords = keywords.sort_index()
    return keywords

def get_stoplist(**kwargs)->list:
    """ gets german stoplist from github repo optionally saves it to file
        saves stoplist if parameter 'filepath' is supplied
    """
    
    stoplist_de = None
    stop_words_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-de/master/stopwords-de.txt"
    filename = os.path.basename(stop_words_url)
    fp = kwargs.get("filepath",None)
    f_ref = None
    if fp is not None:
        f_ref = os.path.join(fp,filename)
        if os.path.isfile(f_ref):
            print(f"Reading stoplist from {f_ref}")
            stopwords = read_file(f_ref)
            return ("").join(stopwords).split("\n")            

    try:
        print(f"Reading stoplist from {stop_words_url}")
        r = requests.get(stop_words_url, allow_redirects=True)
        stoplist_de = r.content.decode('UTF-8').split("\n")        
    except:
        print(f"Exception accessing url {stop_words_url}")
        print(traceback.format_exc())            
    
    # optionally save to file
    if f_ref is not None:        
        print(f"Writing Stoplist to {f_ref}")        
        open(f_ref, 'wb').write(r.content)        
    
    return stoplist_de

def process_keyword_cluster(keyword_cluster:dict,key_len:int=5)->dict:
    entry = {}
    s = 0
    key_s = ""
    for cl_k,cl_n in keyword_cluster.items():
        s += cl_n
        key_s += f"{cl_k}({cl_n}), "        
    count = len(keyword_cluster.values())
    if count == 1:
        entry["key"] = cl_k
    else:
        entry["key"] = cl_k[:key_len]
    entry["count"] = count
    entry["sum"] = s       
    entry["keywords"] = key_s
    entry["keyword_dict"] = keyword_cluster
    return entry

def get_keyword_clusters(keywords:dict,key_len:int=5)->dict:
    """ bundles keywords according to common first letters of keyword """
    last_cluster_key = ""
    keywords_cluster_dict = {}
    keyword_cluster = {}
    last_keyword = ""

    for k,n in keywords.items():
        current_cluster_key = k[:key_len]
        if not current_cluster_key == last_cluster_key:      
            if keyword_cluster:
                entry = process_keyword_cluster(keyword_cluster,key_len)
                keywords_cluster_dict[entry["key"]] = entry
            last_cluster_key = current_cluster_key           
            keyword_cluster = {}
        keyword_cluster[k] = n
        last_keyword = k

    # process last item    
    if not keywords_cluster_dict.get(current_cluster_key,None) and keyword_cluster:
        entry = process_keyword_cluster(keyword_cluster,key_len)
        keywords_cluster_dict[entry["key"]] = entry          
    
    return keywords_cluster_dict

def get_keywordcluster_df(f:str,**kwargs)->pd.DataFrame:
    """ gets a dataframe containing keyword clusters from file given by path f. 
        By default, a stoplist will be used
        optional parameters as kwargs: 
        fp_stoplist: filepath for stoplist (will create one if not found)
        stoplist: use your own stoplist of keywords (words not counted)        
        key_len: key length for clusters
    """
    
    # get the stoplist 
    stoplist = None
    stoplist = kwargs.get("stoplist",None)
    fp_stoplist = kwargs.get("fp_stoplist",None)
    key_len = kwargs.get("key_len",6)
    
    if stoplist is None and fp_stoplist is not None:
        stoplist = get_stoplist(filepath=fp_stoplist)
    
    if isinstance(stoplist,list):
        print(f"Using stoplist with {len(stoplist)} elements")
        
    # get the keyword cluster
    lines = read_file(f)
    print(f"Number of Lines in {f}: {len(lines)}") 
    keywords = get_keywords(lines,stop_words=stoplist)
    print(f"Number of Keywords: {len(keywords)}")         
    keywords_cluster_dict = get_keyword_clusters(keywords,key_len=key_len)
    df = pd.DataFrame.from_dict(keywords_cluster_dict, orient='index')
    df = df.drop(columns=['keyword_dict','key'])
    df = df.sort_values(by='sum', ascending=False)
    print(f"Number of Clustered Keywords: {df.shape[0]}")    
    return df
        
# this will create a CSV with a wordcount from text document    
#keywords.to_csv(f_csv,index=True,sep=";")