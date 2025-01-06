# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:50:10 2025

@author: kashy
"""

'''
Given a list of crops, kinds,
read all the csv files. 
Then, filter by requested countries. 

If any of crops, kinds, countries are null, use all. 
(kinds - Area, Production, Yield.)
'''

import pandas as pd
import glob 
import GlobalVars as gv
import os 





def one_big_file(crops,kinds=['Yield']):
    
    fnames = glob.glob(f'{gv.path_agstat}/*.csv')
    list_of_dfs = []
    for fname in fnames:
        name = os.path.basename(fname)
        split = name.split('_')
        assert len(split) ==2 , f"Filename, 'f{name}' should only have 1 underscore. Found = {len(split)-1}."
        crop, kind = split 
        kind = kind.split('.')[0]
        
        #If crops or kinds are [] , use all. 
        if(crop not in crops and len(crops)!=0):
            continue  
        if(kind not in kinds and len(kinds)!=0):
            continue  
        
        df = pd.read_csv(fname)
        df['crop'] = crop
        df['kind'] = kind 
        list_of_dfs.append(df)
    
    df = pd.concat(list_of_dfs, ignore_index=True)
    
    return df 


def filter_countries(onebigfile, countries):
    if(len(countries)!=0):
        onebigfile = onebigfile[onebigfile['ADM0_NAME'].isin(countries)]
    return onebigfile
     
        
def get_specific(df , crop,kind,country):
    df = df[df['crop']==crop]
    df = df[df['ADM0_NAME'] == country]
    df = df[df['kind'] == kind]
    return df 

        
    

if __name__ == '__main__':
    crops = []
    kinds = []
    countries = ['Ukraine','Russia']
    df = one_big_file(crops,kinds)
    df = filter_countries(df,countries)
    
    df.to_csv(gv.path_agstat+'/merged/merged.csv')
    
    '''
    # For one crop and one kind, assert that ADM2_NAME is unique
    crops = ['WinterWheat']
    df = one_big_file(crops) #kinds default is yield
    ad2 = df['ADM2_NAME'].dropna()
    assert ad2.is_unique , "ADM2_NAME is not unique"
    '''
    
    
    
    


