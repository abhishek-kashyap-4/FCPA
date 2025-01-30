


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
Since we want the code to work for both single country and global, 
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

import warnings 

def get_input_cropland(path):
    try:
        #with rio.open(path) as src:
         #   data = src.read(1)
        #return data
        src = rio.open(path)
        return src
    except Exception as e:
        raise FileNotFoundError("Check if country, Crop are available.")

def get_country_zones(country, zone_level,gaul=True):
    
    #print(gv.path_admin_zones , country , zone_level)
    if(gaul):
        #Use the gaul dataset. 
        filepath= gv.path_admin_zones+'/gaul/Zone'+str(zone_level)
    else:
        filepath= gv.path_admin_zones+'/'+country+'/Zone'+str(zone_level)
        
    fnames = glob.glob(f'{filepath}/*.shp')
    assert len(fnames)==1 , f"Filepath - {filepath} needs to have 1 shp file. found {len(fnames)}"
    gdf = gpd.read_file(fnames[0])
    
    if(gaul):
        #If gaul, filter the dataset by country
        gdf = gdf[gdf.name0 == country]
        assert len(gdf)>0 , f"Empty gdf returned, check the countryname , <<{country}>> properly."
    else:
        if(zone_level == 1):
            print(gdf.columns)
            gdf['name1'] = gdf['ADM1_EN']
    return gdf
    
    
    
    


#Get all agstats
agstat_file = agstat.one_big_file(crops=[],kinds=[])
#agstat.get_specific(agstat_file, crop=crop, kind=kind, country=country)

def execute(country, crop, year ):
    
    
    '''
    
    '''
if __name__ == '__main__':
    for country in gv.countries:
        for crop in gv.crops:
            # Input crop mask 
            path = gv.input_crop_map[crop]
            src = get_input_cropland(path)
            #plt.imshow(img,cmap='pink')
            
            
            #Get country geometry 
            country_adm0 = get_country_zones(country,zone_level=0,gaul=False)#.iloc[0].geometry
            warnings.warn("other admin zone shapefiles are still being used. \n Note - this behaviour can be changed since rmask can take multiple shapes. you don't need admin0 level data at all. ")
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
            
            
            
            warnings.warn("When you get admin_zones of a specific level, make sure you don't get zones of a higher division. Because when you fetch admin1, you want to analyze admin 1. You don't want admin 2 geography.")
            #Get admin 1 zones of the country. 
            country_adm1 = get_country_zones(country,zone_level=1)
            assert country_adm1.crs == src.crs , "CRS mismatch, reproject."
            country_adm1['name1_SIMP'] = country_adm1['name1'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
            
            #Agstats 
            stat = agstat.get_specific(agstat_file, crop=crop, country=country,kind='Yield')
            stat['ADM1_NAME_SIMP'] = stat['ADM1_NAME'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
            
            assert set(country_adm1['name1_SIMP']) == set(stat['ADM1_NAME_SIMP'])
            warnings.warn("The names of admin zones aren't consistent across admin_zones and agstats. I am currently merging based on all lower alphabet characters.")
            
            merged_z1 = pd.merge(stat, country_adm1,how='left', left_on='ADM1_NAME_SIMP', right_on='name1_SIMP')
            
            "!IMP! When you merge by adm1 , you will get adm1 geometry for all adm2,3,4 (right?)"
            #filter based on ADM2_NAME being null 
            #I don't understand why I would do this.
            merged_z1 = gpd.GeoDataFrame(merged_z1[merged_z1['ADM2_NAME'].isna()])
            print(merged_z1.geometry)
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
            '''
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
                            
                            '''
            
            
            
            
            
            
            
                    
            
            
    
