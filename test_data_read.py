# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:33:05 2025

@author: kashy
"""

from geoprepare.stats import geom_extract


import re
import numpy as np
import pandas as pd
import rasterio as rio 
from rasterio.mask import mask as rmask
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import glob

import agstat
import GlobalVars as gv

import warnings 
from itertools import chain


from main import get_country_zones , get_input_cropland 

crop = 'WinterWheat'
country = 'Ukraine'

def get_indicator(year,doy,n=5):
    # everyday doesn't need to be present, maybe select among n nearest days
    for increment in  chain(range(n), range(-n, 0)):
        try:
            dataset = rio.open('Data/MODIS/ndvi/mod09.ndvi.global_0.05_degree.'+str(year)+'.'+str(doy)+'.c6.v1.tif')
            return dataset
        except :
            pass 
    raise Exception(f"No proper date found in year {year}; doy range from {doy-n} to {doy+n}")
    

path = gv.input_crop_map[crop]
src = get_input_cropland(path)



indicator = get_indicator(year = 2010 , doy = 337)


country_adm0 = get_country_zones(country,zone_level=0,gaul=False)
country_geom = country_adm0.iloc[0].geometry



limit = 20 


val_extracted = geom_extract(
    geometry = country_geom,
    variable = 'ndvi',
    indicator = indicator,
    stats_out=["mean", "counts"],
    afi=src,
    afi_thresh=limit * 100,
    thresh_type="Fixed"
)