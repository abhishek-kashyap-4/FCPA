


# Total script for FPCA

#Take - 
    #1. Admin (1,2) level agricultural statistics across years 
    #2. Initial global cropmask for spring wheat, winter wheat 
    #3. Vector datasets that define which grid belongs to which admin zone. 
    
    #4. Satellite Data across the years for machine learning of 5x5 km (same as global cropmask)
    
# Use - 
    #1. Global variables that have years, crops to loop through. 
    #2. 
'''
Since we want th ecode to work for both single country and global, 
Take as inputs vector datasets with desired admin level. 
Tiff files should be created within these zones. 

'''

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

def get_input_cropland(path):
    try:
        #with rio.open(path) as src:
         #   data = src.read(1)
        #return data
        src = rio.open(path)
        return src
    except Exception as e:
        raise FileNotFoundError("Check if country, Crop are available.")

def get_country_zones(country, zone_level):
    #print(gv.path_admin_zones , country , zone_level)
    filepath= gv.path_admin_zones+'/'+country+'/Zone'+str(zone_level)
    fnames = glob.glob(f'{filepath}/*.shp')
    assert len(fnames)==1 , f"Filepath - {filepath} needs to have 1 shp file. found {len(fnames)}"
    gdf = gpd.read_file(fnames[0])
    return gdf
    
    
    
    


#Get all agstats
agstat_file = agstat.one_big_file(crops=[],kinds=[])
#agstat.get_specific(agstat_file, crop=crop, kind=kind, country=country)

def execute(country, crop, year ):
    
    
    '''
    
    '''

for country in gv.countries:
    for crop in gv.crops:
        # Input crop mask 
        path = gv.input_crop_map[crop]
        src = get_input_cropland(path)
        #plt.imshow(img,cmap='pink')
        
        #Get country geometry 
        country_adm0 = get_country_zones(country,zone_level=0)#.iloc[0].geometry
        assert country_adm0.crs == src.crs , "CRS mismatch, reproject."
        country_geom = country_adm0.iloc[0].geometry

        #Filter tiff file based on country_geom
        out_image, out_transform = rmask(src, list(country_geom.geoms), crop=True)
        out_meta = src.meta
        # Update the metadata for the new cropped raster
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        #plt.imshow(out_image[0,:,:],cmap='pink')
        with rio.open(gv.path_tiff_output+f'//{country}_{crop}_generic.tif', "w", **out_meta) as dest:
            dest.write(out_image)
        
        
        
        #Get admin 1 zones of the country. 
        country_adm1 = get_country_zones(country,zone_level=1)
        assert country_adm1.crs == src.crs , "CRS mismatch, reproject."
        country_adm1['ADM1_EN_SIMP'] = country_adm1['ADM1_EN'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
        
        #Agstats 
        stat = agstat.get_specific(agstat_file, crop=crop, country=country,kind='Yield')
        stat['ADM1_NAME_SIMP'] = stat['ADM1_NAME'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
        
        merged_z1 = pd.merge(stat, country_adm1,how='left', left_on='ADM1_NAME_SIMP', right_on='ADM1_EN_SIMP')
        #filter based on ADM2_NAME being null 
        merged_z1 = gpd.GeoDataFrame(merged_z1[merged_z1['ADM2_NAME'].isna()])
        #print(type(gpd.GeoDataFrame(merged_z1)))
        
        
        for year in gv.years:
            m = merged_z1[[str(year),'geometry']]
            
        ##//////////
        with rio.open(gv.path_tiff_output+f'//{country}_{crop}_generic.tif') as src:
            raster_data = src.read(1)  # Read the first band (assumes single-band raster)
            raster_transform = src.transform  # Affine transform for the raster
            raster_crs = src.crs
            raster_meta = src.meta
        
        # Ensure CRS alignment
        merged_z1 = merged_z1.to_crs(raster_crs)
        1/0
        for year in gv.years:
             
            for row in range(raster_data.shape[0]):
                for col in range(raster_data.shape[1]):\
    
                    # Get the pixel's center coordinates
                    x, y = rio.transform.xy(raster_transform, row, col, offset="center")
                    pixel_point = Point(x, y)
                    
                    # Find the admin zone containing this point
                    zone = merged_z1[merged_z1.contains(pixel_point)]
                    if not zone.empty:
                        assert len(zone)==1 
                        yield_val = zone.iloc[0][str(year)]
                        print(yield_val)
                        raster_data[row, col] = yield_val
                    else:
                        raster_data[row, col] = raster_meta['nodata']
            
            

            # Save the adjusted raster
            raster_meta.update(dtype="float32")  # Update metadata for new data type
            with rio.open(gv.path_tiff_output+f'//{country}_{crop}_{year}.tif', "w", **raster_meta) as dst:
                dst.write(raster_data, 1)  # Write the adjusted data to band 1
                        
                        
        
        
        
        
        
        
        
                
        
        

