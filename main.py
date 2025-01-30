


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
print('Please go through the comments inthe code once. ')


import re
import numpy as np
import pandas as pd
import rasterio as rio 
from rasterio.mask import mask as rmask
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import glob
from itertools import chain

import agstat
import GlobalVars as gv

import warnings 


def get_indicator(year,doy,variable = 'ndvi',n=5):
    warnings.warn('You really should get all DOY of a year and aggregate accordingly.')
    # everyday doesn't need to be present, maybe select among n nearest days
    for increment in  chain(range(n), range(-n, 0)):
        try:
            dataset = rio.open('Data/MODIS/'+variable+'/mod09.'+variable+'.global_0.05_degree.'+str(year)+'.'+str(doy+increment)+'.c6.v1.tif')
            return dataset
        except :
            pass
    raise Exception(f"No proper date found in year {year}; doy range from {doy-n} to {doy+n}")
    
    
    
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
    
    
    
    



from geoprepare.stats import geom_extract

def gatherer(roi , input_raster, indicator, variable, year):
    '''
    Given a region of interest , 
        use geoprepare to get the indicator variables for the input raster.
    This will be looped over differed ROI of a country. 
    This function will return a row  corresponding to
        a single roi(admin zone) , 
        a single timeperiod,
        all the required variables. 
    This will be 1 row in the training data. 
        
    '''
    #print('IMPORTANT Behaviour has to change. Loop for variables has to be before roi, not here. also 1 variable = 1 indicator.')
    #this now changed. 
    output= {}
    limit = 20 
        
    val_extracted = geom_extract(
        geometry = roi,
        variable = variable,
        indicator = indicator,
        stats_out=["mean", "counts"],
        afi=input_raster,
        afi_thresh=limit * 100,
        thresh_type="Fixed"
    )
    stats = val_extracted['stats']
    
    output |= {variable+'__'+key:stats[key] for key in stats }
    return output 



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
    
def learner(df):
   df = df.dropna()
   X = df.drop(columns=['Yield'])
   y = df['Yield']
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
   
   model = RandomForestRegressor(n_estimators=100)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   loss = mean_squared_error(y_test, y_pred)
   return model

from rasterio.mask import mask
from shapely.geometry import mapping

buddi = 0

def predictor(model , raster,roi):
    '''
    Given the trained model and raster, use the model on the raster and save the raster.
    roi is now the country's roi
    '''
    global buddi 
    img, transform = mask(raster, [mapping(roi)], crop=False, nodata=None)
    meta = raster.meta.copy()


    img = np.moveaxis(img, 0, -1)
    flat_pixels = img.reshape(-1, img.shape[2])
    flat_pixels = np.nan_to_num(flat_pixels)
    
    predictions = model.predict(flat_pixels)

    result_raster = np.full((img.shape[0], img.shape[1]), np.nan)

    buddi = pd.DataFrame(flat_pixels)
    buddi['predictions'] = predictions 
    123/0
    valid_mask = ~np.isnan(img).any(axis=2)
    result_raster[valid_mask] = predictions

    meta.update(dtype=rio.float32, count=1)


    return result_raster

        
def binder(merged,input_raster,year,variables = ['ndvi'] ):
    '''
    Take all the geometries of admin zones. You are not concerned with zone level anymore. 
    Return a dataframe that can be ML trained. 
    You perhaps will be taking the merged file here with will have aqgstats
    
    '''

    rows = []
    for variable in variables:
        single_column = []
        #1. Get the indicator. For now, get a random DOY;
            # but understand that this may change in the future.  
        indicator = get_indicator(year = year , doy = 100 , variable = variable , n = 50)
        print("Here you will merge your indicators into a single image.")
        
        # check for the case where there is no geometry, since merged has a left join. 
        for geom in merged['geometry']: #It is assumed that you have the respective admin zone geometry. 
            single_column.append( gatherer(geom , input_raster ,indicator , variable ,year))
            # If this gives error , append empty row.
        rows.append(single_column)
    
    rows = [pd.DataFrame(row) for row in rows]
    df = pd.DataFrame(pd.concat(rows,axis=1))
    warnings.warn("Please check the concat axis once you have multiple indicators.")
    
    assert len(set(merged['kind'])) == 1  and merged['kind'].iloc[0] == 'Yield'
    df['Yield'] = merged[str(year)]
    
    model = learner(df)
    
    warnings.warn("You are supposed to merge all the indicators used input a single image first.")
    #Get country geometry 
    country_adm0 = get_country_zones(country,zone_level=0,gaul=False)#.iloc[0].geometry
    warnings.warn("other admin zone shapefiles are still being used. ")
    assert country_adm0.crs == src.crs , "CRS mismatch, reproject."
    country_geom = country_adm0.iloc[0].geometry
    output_raster = predictor(model , indicator ,country_geom )

    out_meta = src.meta
    # Update the metadata for the new cropped raster
    out_meta.update({
        "driver": "GTiff",
        #"height": output_raster.shape[0],
        #"width": output_raster.shape[1],
        "transform": input_raster.transform 
    })
    #plt.imshow(out_image[0,:,:],cmap='pink')
    
    with rio.open(gv.path_tiff_output+f'//{country}_{crop}_ML.tif', "w", **out_meta) as dest:
        dest.write(output_raster.astype(rio.float32), 1)
        
    
    
    

def executer(country, crop, year ):
    
    
    '''
    
    '''
if __name__ == '__main__':
    
    #Get all agstats
    agstat_file = agstat.one_big_file(crops=[],kinds=[])
    #agstat.get_specific(agstat_file, crop=crop, kind=kind, country=country)
    year = 2015
    warnings.warn("I am assuming the year is 2015 for the input cropmask. the cropmask is for 2015-2020. I have to see if I need to add all these years to the training data.")
    
    
    for country in gv.countries:
        for crop in gv.crops:
            # Input crop mask 
            path = gv.input_crop_map[crop]
            src = get_input_cropland(path)
            #plt.imshow(img,cmap='pink')
            
            '''
            #Get country geometry 
            country_adm0 = get_country_zones(country,zone_level=0,gaul=False)#.iloc[0].geometry
            warnings.warn("other admin zone shapefiles are still being used. \n Note - this behaviour can be changed since rmask can take multiple shapes. you don't need admin0 level data at all. UPDATE - you don't even need rmask. ")
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
            '''
            
            
            warnings.warn("When you get admin_zones of a specific level, make sure you don't get zones of a higher division. Because when you fetch admin1, you want to analyze admin 1. You don't want admin 2 geography.")
            #Get admin 1 zones of the country. 
            country_adm1 = get_country_zones(country,zone_level=1)
            assert country_adm1.crs == src.crs , "CRS mismatch, reproject."
            country_adm1['name1_SIMP'] = country_adm1['name1'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
            
            #Agstats 
            stat = agstat.get_specific(agstat_file, crop=crop, country=country,kind='Yield',zone=1)
            stat['ADM1_NAME_SIMP'] = stat['ADM1_NAME'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
            
            assert set(country_adm1['name1_SIMP']) == set(stat['ADM1_NAME_SIMP'])
            warnings.warn("The names of admin zones aren't consistent across admin_zones and agstats. I am currently merging based on all lower alphabet characters.")
            
            merged_z1 = pd.merge(stat, country_adm1,how='left', left_on='ADM1_NAME_SIMP', right_on='name1_SIMP')
            
            "!IMP! When you merge by adm1 , you will get adm1 geometry for all adm2,3,4 (right?)."
            " no , you will get duplicate rows"
            #filter based on ADM2_NAME being null 
            #I don't understand why I would do this.
            # i do now suckaaa. but this is now being done separately. 
            merged_z1 = gpd.GeoDataFrame(merged_z1[merged_z1['ADM2_NAME'].isna()])
            

            merged_z1['ADM_rel_NAME'] = merged_z1['name1_SIMP']
            #m = merged_z1[['ADM_rel_NAME',str(year),'geometry']]
            
                
            # Ensure CRS alignment
            merged_z1 = merged_z1.to_crs(src.crs)
            binder(merged_z1,input_raster=src,year=year )
            
            
            
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
            
            
            
print("Research about whether storing rio in memory or making rasterio reads is better.")
            
            
            
                    
            
            
    
