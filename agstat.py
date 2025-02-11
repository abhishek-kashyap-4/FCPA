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
import re 


import warnings




def one_big_file(crops,kinds=['Yield','Area','Production']):
    
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
     
def get_specific_wrap(df , crop , kind , countrylist , zone):
    dfs = []
    for country in countrylist:

        dfs.append(get_specific(df , crop , kind = kind , country = country , zone = zone))
    df = pd.concat(dfs,axis=0)
    return df 
def get_specific(df , crop,kind,country,zone=1):
    df = df[df['crop']==crop]
    if(country == 'Russia'):
        warnings.warn("This needs to be enhances in get_specific naming convention")
        country1 = 'Russian Federation'
        df = df[df['ADM0_NAME'] == country1]
        df['ADM0_NAME'] = df['ADM0_NAME'].replace(country1,country) 
        
    else:
        df = df[df['ADM0_NAME'] == country]
    if(kind !='all'):
        df = df[df['kind'] == kind]
    else:
        #Get single columns, change years to <<kind>>_<<year>>
        years  = [col for col in df.columns if re.search(r'^[0-9]+$',col)]
        others = [col for col in df.columns if not re.search(r'^[0-9]+$',col)]
        base = df[df.kind == 'Yield'][others]
        base.reset_index(inplace=True,drop=True)
        
        for idx,grp in df.groupby('kind'):
            grp = grp[years].rename(columns = {y: idx+'_'+y for y in years})
            grp.reset_index(inplace=True,drop=True)
            base = pd.concat([base , grp],axis=1)
        df = base

    
        
    #given zone >=0 , delete all the rows where zone is null zone+1 are all not nulls.
    # zone +1 null mean zone +n are all nulls, based on how the data is. 
    if(zone == 0):
        df = df[df['ADM0_NAME'].notnull()]
        #adm1 null automatically means adm2 null. 
        df = df[df['ADM1_NAME'].isnull()]
        assert len(df) ==1 , f"Exactly 1 country should match, found {len(df)}"
    elif(zone == 1):
        df = df[df['ADM1_NAME'].notnull()]
        df = df[df['ADM2_NAME'].isnull()]
        assert len(df) >0 , "No rows found."
        
    elif(zone == 2):
        df = df[df['ADM2_NAME'].notnull()] 
        assert len(df) >0 , "No rows found."
    else:
        raise NotImplementedError 
    
    return df 

        
    

if __name__ == '__main__':
    crops = []
    kinds = []
    countries = ['Ukraine','Russia']
    df = one_big_file(crops,kinds)
    #df = filter_countries(df,countries)
    agstat = get_specific(df, crop='WinterWheat', country='Ukraine',kind='all',zone=1)
            
    
    df.to_csv(gv.path_agstat+'/merged/merged.csv')
    
    '''
    # For one crop and one kind, assert that ADM2_NAME is unique
    crops = ['WinterWheat']
    df = one_big_file(crops) #kinds default is yield
    ad2 = df['ADM2_NAME'].dropna()
    assert ad2.is_unique , "ADM2_NAME is not unique"
    '''
    
    
    
    


