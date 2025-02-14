


# Total script for FPCA

#Take - 
    #1. Admin (1,2) level agricultural statistics across years 
    #2. Initial global cropmask for spring wheat, winter wheat 
    #3. Vector datasets that define which grid belongs to which admin zone. 
    
    #4. Satellite Data across the years for machine learning of 5x5 km (same as global cropmask)
    
# Use - 
    #1. Global variables that have years, crops to loop through. 
    #2. 
import logging
logging.basicConfig(filename="logger.log",
                    level=logging.INFO,
                    format="%(asctime)s --- %(message)s") 
'''
# Example log messages at different severity levels
logging.debug("This is a DEBUG message.")
logging.info("This is an INFO message.")
logging.warning("This is a WARNING message.")
logging.error("This is an ERROR message.")
logging.critical("This is a CRITICAL message.")
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
'''

logging.info('We want the code to work for single countries as well as several/global. \
             ')

print('Please go through the comments in the code once. ')



import re
import numpy as np
import pandas as pd
import rasterio as rio 
from rasterio.mask import mask

from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import glob
from itertools import chain

import agstat
import GlobalVars as gv
import glob

import warnings 



def get_indicator_agg(year ,roi, variable = 'ndvi',agg = 'max'):
    
    
    fnames = glob.glob(gv.path_tiff_input+'\\'+variable.upper()+'/*.tif')
    fnames = [name for name in fnames if '0.05_degree.'+str(year) in name]
    fnames = sorted(fnames)
    rasters = []
    for fname in fnames:
        fdata = rio.open(fname)
        
        'After the code changes, roi is supposed to be a list already'
        fdata , transform = mask(fdata, roi, crop=True)
        logging.warning('In get_indicator_agg, we are assuming indicator files are single band rasters.')
        #IMP assuming it is a single band raster
        rasters.append(fdata[0])
        
    stack = np.stack(rasters, axis=0)  # Shape: (num_rasters, height, width)
    if(agg == 'max'):
        aggregated_array = np.max(stack, axis=0)  # Apply aggregation along the first axis 
    elif(agg == 'mean'):
        aggregated_array = np.mean(stack, axis=0)  # Apply aggregation along the first axis 
    else:
        raise NotImplementedError
    
    with rio.open(fnames[0]) as src:
        profile = src.profile
        meta = src.meta
        meta.update({
        "driver": "GTiff",
        "height": fdata.shape[1],
        "width": fdata.shape[2],
        "transform": transform,
        "crs": GLOBAL_CRS
        })
    src.close()
        
    path = gv.path_tiff_interim +f'\\Raster__{variable}_{agg}.tif'
    
    profile.update(dtype=rio.float32, count=1)
    with rio.open(path, "w", **meta) as dst:
        dst.write(aggregated_array.astype(rio.float32), 1)
    dst.close()
    return path
            
        
    

def get_indicator(year,doy,variable = 'ndvi',n=5):
    assert variable == 'ndvi' , f"Not implemented variable {variable}"
    # everyday doesn't need to be present, maybe select among n nearest days
    for increment in  chain(range(n), range(-n, 0)):
        try:
            dataset = rio.open('Data/MODIS/'+variable+'/mod09.'+variable+'.global_0.05_degree.'+str(year)+'.'+str(doy+increment)+'.c6.v1.tif')
            return dataset
        except :
            pass
    raise Exception(f"No proper date found in year {year}; doy range from {doy-n} to {doy+n}")
    

from rasterio.merge import merge

def merge_raster_bands(raster_paths, output_path):
    """
    Merges all bands from multiple rasters and writes them to a single raster file.
    
    Parameters:
        raster_paths (list of str): List of file paths to input rasters.
        output_path (str): File path for the output combined raster.
    """
    all_bands = []
    first_meta = None

    for i, raster_path in enumerate(raster_paths):
        with rio.open(raster_path) as src:
            # Read all bands from the current raster
            bands = [src.read(band + 1) for band in range(src.count)]
            all_bands.extend(bands)  # Append to the list of all bands
            
            # Capture metadata from the first raster
            if i == 0:
                first_meta = src.meta.copy()
    
    # Update metadata for the output raster
    out_meta = first_meta
    out_meta.update({
        "count": len(all_bands),  # Total number of bands
        "dtype": all_bands[0].dtype,  # Assuming all bands have the same data type
    })

    # Write the combined raster to a new file
    with rio.open(output_path, "w", **out_meta) as dest:
        for i, band in enumerate(all_bands):
            dest.write(band, i + 1)
    return output_path



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
            gdf['name1'] = gdf['ADM1_EN']
    return gdf
    
def get_country_zones_new(regionlist, zone_level,gaul=True):
    
    #print("IMPLEMENT THIS")
    #return get_country_zones(country=regionlist[0], zone_level=zone_level,gaul=gaul)
    
    if(gaul):
        warnings.warn("This needs to be modified")
        if('Russia' in regionlist):
            regionlist.append('Russian Federation')
        #Use the gaul dataset. 
        filepath= gv.path_admin_zones+'/gaul/Zone'+str(zone_level)
        fnames = glob.glob(f'{filepath}/*.shp')
        assert len(fnames)==1 , f"Filepath - {filepath} needs to have 1 shp file. found {len(fnames)}"
        gdf = gpd.read_file(fnames[0])
        #If gaul, filter the dataset by regionlist
        gdf = gdf[gdf.name0.isin(regionlist)]
        assert len(gdf)>0 , f"Empty gdf returned, check the regions , <<{regionlist}>> properly."
    else:
        
        filepaths= [gv.path_admin_zones+'/'+country+'/Zone'+str(zone_level) for country in regionlist]

        fnames = [glob.glob(f'{filepath}/*.shp') for filepath in filepaths]
        assert len(fnames)>0 , "No files found"
        "This assert doesn't work because of the following commnet"
        #glob creates lists of list for some reason 
        gdf = pd.concat([gpd.read_file(fname[0]) for fname in fnames])
        if(zone_level == 1):
            gdf['name1'] = gdf['ADM1_EN']
            
    return gdf    
    
    
from geoprepare.stats import geom_extract
def gatherer(roi , input_raster_path, indicator, variable, year,agg_name = ''):

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
    
    with rio.open(input_raster_path) as input_raster:
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
    
    # Use the aggregation that was used on the indicator as a prefix.
    output |= {agg_name + variable+'__'+key:stats[key] for key in stats }
    '''except Exception as e:
        print(f"Handling gatherer exception: {e}")
        output = {agg_name + variable + '__mean': None}
        warnings.warn('DELICATE')'''
    return output 

#%%


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ml , ref = 0 , 0
def diagnostics(output_raster_path, input_raster_path ,country_geom, gdf):
    global ml 
    global ref 
    output_raster = rio.open(output_raster_path)
    input_raster = rio.open(input_raster_path)
    
    "country geom  will already be a list"
    ref , transform = mask(input_raster, country_geom, crop=True)
    
    ml_flat = output_raster.read(1).flatten()
    ref_flat = ref.flatten()
    ml = ml_flat 
    ref = ref_flat
    
    ref_flat = ref_flat.astype('float64')
    ref_flat /= 100 * 100 #unit conversion.
    
    
    mask1= (ml_flat != output_raster.nodata) & (ref_flat != input_raster.nodata)
    ml_flat = ml_flat[mask1]
    ref_flat = ref_flat[mask1]
    # Calculate metrics
    mae = mean_absolute_error(ref_flat, ml_flat)
    rmse = np.sqrt(mean_squared_error(ref_flat, ml_flat))
    r2 = r2_score(ref_flat, ml_flat)
    print(mae,rmse , r2)
    
    
    
def plots(df , xfeature , yfeature):
    sns.scatterplot(data = df,x =xfeature , y= yfeature )
    #plt.scatter(vals,model.predict(vals))
    plt.show()

chekka1 = 0
def learner(df,ylabel):
   global chekka1
   global poly 
   df = df.dropna()
   X = df.drop(columns=[ylabel])
   y = df[ylabel]
   
   scaler = StandardScaler()
   #X_scaled = scaler.fit_transform(X)
   #poly = PolynomialFeatures(degree=2)
   y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
   
   chekka1 = y
   X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2)
   
   model = RandomForestRegressor(n_estimators=2)
   #model = LinearRegression()
   #model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   loss = mean_squared_error(y_test, y_pred)
   print(df)
   '''
   for col in X.columns:
       #plots(df , col , 'Area')
       sns.scatterplot(data = df,x =col , y= 'Area' )
       plt.show()
    '''
       
   print(loss)
   return model

chekka = 0
def predictor(model , raster_path,roi,fill_value=0.0):
    global chekka
    "CHANGE LATER"
    roi = roi.iloc[0]
    '''
    Given the trained model and raster, use the model on the raster and save the raster.
    roi is now the country's roi
    roi is now a list of country rois
    '''
    
    
    with rio.open(raster_path) as raster:
        #img, transform = mask(raster, [roi], crop=True)
        img = raster.read()
        transform = raster.transform
        meta = raster.meta.copy()
        
    valid_mask = img[0] != raster.nodata 
    assert fill_value   == raster.nodata , "Once predicted, make sure to use the correct nodata"
    # Assuming first band can be used for masking
    
    
    X = img[:, valid_mask].T  # Flatten valid pixels to (pixels, bands)
    predictions = model.predict(X)
    result_raster = np.full(img.shape[1:], fill_value,  dtype=np.float32)
    result_raster[valid_mask] = predictions
    
    
    chekka = predictions
    meta.update({"dtype": "float32", "height": img.shape[1], "width": img.shape[2], "transform": transform})
    
        
    return result_raster , meta


def get_output(gdf, ylabel, year, indicator_path , country_geom , how = 'ml',output_path=gv.path_tiff_output ):
    '''
    Input - GeoDataFrame used for training 
    Output -  Raster output path 
    '''
    
    if(how == 'version1'):
        pass 
    elif(how == 'version2'):
        pass 
    elif(how == 'ml'):
        
        model = learner(gdf,ylabel = ylabel)
        output_path += '\\Raster_ML'
        output_raster , meta = predictor(model , indicator_path ,country_geom)

    else:
        raise NotImplementedError 
    
    
    output_path += '.tif'
    print(meta)
    with rio.open(output_path, "w", **meta) as dest:
        dest.write(output_raster, 1)
        
    warnings.warn("Change this once it is not a single band raster")
    return output_path
    
#%%



#%%

def binder(merged,input_raster_path,year,country_adm0,agg_variables = ['max_ndvi','mean_ndvi'],ylabel = 'Area' ):
    
    '''
    Take all the geometries of admin zones. You are not concerned with zone level anymore. 
    Return a dataframe that can be ML trained. 
    You perhaps will be taking the merged file here with will have aqgstats
    
    '''
    
    
    logging.warning("other admin zone shapefiles are still being used. ")
    #with rio.open(input_raster_path) as input_raster:
     #   assert country_adm0.crs == input_raster.crs , "CRS mismatch, reproject."
        
    assert country_adm0.crs == GLOBAL_CRS , "CRS mismatch, reproject."
    
    country_geom = country_adm0.geometry
    

    rows = []
    logging.info('Standalone (unaggregated) variables are not implemented yet.')
    indicators = []
    for aggvar in agg_variables:
        agg , variable = aggvar.split('_')
        single_column = []
        #1. For standalone indicartor get a random DOY;
        #indicator = get_indicator(year = year , doy = 100 , variable = variable , n = 50)
        indicator = get_indicator_agg(year = year , roi =country_geom , variable = variable ,agg=agg)
        indicators.append(indicator)
        #total_areas = []
        # check for the case where there is no geometry, since merged has a left join. 
        #for geom in merged['geometry']: #It is assumed that you have the respective admin zone geometry. 
        for idx,row in merged.iterrows():
            geom = row['geometry']
            print(geom.to_crs("EPSG:6933")[0].area/1e4)
            #total_areas.append(geom.area)
            single_column.append( gatherer(geom , input_raster_path ,indicator , variable ,year,agg_name = agg))
            #you need to get the indicatory copy
            # If this gives error , append empty row.

        rows.append(single_column)
        
    rows = [pd.DataFrame(row) for row in rows]
    
    df = pd.DataFrame(pd.concat(rows,axis=1))
    
    
    ptbl = df.copy()
    ptbl['Yield'] = merged['Yield_'+str(year)]
    
    
    for col in ptbl.columns:
        sns.scatterplot(data = ptbl , x =col , y = 'Yield' )
        plt.show() 
    
    #df[ylabel] = merged[str(year)]
    df[ylabel] = merged[ylabel+'_'+str(year)] / (merged['km2_tot'] * 100)
    print(df)
    output_path = gv.path_tiff_interim + '\\Raster_merged.tif'
    indicator_path = merge_raster_bands(indicators, output_path)
    
    output_raster = get_output(df ,ylabel = ylabel ,year = year , indicator_path = indicator_path  , country_geom = country_geom )
    diagnostics(output_raster ,input_raster_path , country_geom,df )
    
    
    
    

def executer(country, crop, year ):
    
    
    '''
    
    '''
if __name__ == '__main__':
    
    
     
    #Get all agstats
    agstat_file = agstat.one_big_file(crops=[],kinds=[])
    #agstat.get_specific(agstat_file, crop=crop, kind=kind, country=country)
    year = 2015
    logging.warning("I am assuming the year is 2015 for the input cropmask. the cropmask is for 2015-2020. I have to see if I need to add all these years to the training data.")
    ylabel = 'Area' #Also known as kind
    country = gv.countries
    for crop in gv.crops:
        # Input crop mask 
        path = gv.input_crop_map[crop]
        print(path)
        try:
            with rio.open(path) as src:
                GLOBAL_CRS = src.crs
        except:
            raise FileNotFoundError("Check if country, Crop are available.")
        src.close()
        #Get country geometry 
        country_adm0 = get_country_zones_new(country,zone_level=0,gaul=False)#.iloc[0].geometry\
            
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
        
        
        logging.info("When you get admin_zones of a specific level, make sure you don't get zones of a higher division. Because when you fetch admin1, you want to analyze admin 1. You don't want admin 2 geography.")
        #Get admin 1 zones of the country. 
        country_adm1 = get_country_zones_new(country,zone_level=1)
        assert country_adm1.crs == GLOBAL_CRS , "CRS mismatch, reproject."
        country_adm1['name1_SIMP'] = country_adm1['name1'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
        
        #Agstats 
        stat = agstat.get_specific_wrap(agstat_file, crop=crop, countrylist=country,kind='all',zone=1)
        stat['ADM1_NAME_SIMP'] = stat['ADM1_NAME'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
        
        warnings.warn('Ignoring import assert now because not all admin zones (l1) have agstats')
        #assert set(country_adm1['name1_SIMP']) == set(stat['ADM1_NAME_SIMP'])
        logging.info("The names of admin zones aren't consistent across admin_zones and agstats. I am currently merging based on all lower alphabet characters.")
        
        logging.warning("merging is being done with inner now because of spelling mismatches in agstatand GAUL")
        merged_z1 = pd.merge(stat, country_adm1,how='inner', left_on='ADM1_NAME_SIMP', right_on='name1_SIMP')
        
        "!IMP! When you merge by adm1 , you will get adm1 geometry for all adm2,3,4 (right?)."
        " no , you will get duplicate rows"
        #filter based on ADM2_NAME being null 
        #I don't understand why I would do this.
        # i do now suckaaa. but this is now being done separately. 
        # so does it mean this can be removed?
        merged_z1 = gpd.GeoDataFrame(merged_z1[merged_z1['ADM2_NAME'].isna()])

        merged_z1['ADM_rel_NAME'] = merged_z1['name1_SIMP']
        #m = merged_z1[['ADM_rel_NAME',str(year),'geometry']]
        
            
        # Ensure CRS alignment
        merged_z1 = merged_z1.to_crs(GLOBAL_CRS)
        agg_variables = ['max_ndvi','mean_ndvi' , 'max_gcvi' ,'mean_gcvi']
        binder(merged_z1,input_raster_path =path,year=year,ylabel=ylabel , country_adm0=country_adm0,agg_variables = agg_variables )
        
        
        

        
        
                
        
        

