


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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import glob
from itertools import chain

import agstat
import GlobalVars as gv
import glob

import warnings 

#matplotlib.use('Agg')
SUPRESS_VERBOSE = True

logging.warning('In get_indicator_agg, we are assuming indicator files are single band rasters.')


def get_indicator_agg(year ,roi, variable = 'ndvi',agg = 'max'):
    
    
    fnames = glob.glob(gv.path_tiff_input+'\\'+str(year)+'\\'+variable.upper()+'/*.tif')
    fnames = [name for name in fnames if '0.05_degree.'+str(year) in name]
    fnames = sorted(fnames)
    rasters = []
    for fname in fnames:
        fdata = rio.open(fname)
        
        'After the code changes, roi is supposed to be a list already'
        fdata , transform = mask(fdata, roi, crop=True)
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

    
def get_country_zones(regionlist, zone_level,gaul=True):

    if(gaul):
        
        if('Russia' in regionlist):
            warnings.warn("This needs to be modified")
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




from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ml , ref = 0 , 0
def diagnostics(output_raster_path, input_raster_path ,country_geom):
    global ml 
    global ref 
    output_raster = rio.open(output_raster_path)
    input_raster = rio.open(input_raster_path)
    
    "country geom  will already be a list"
    ref , transform = mask(input_raster, country_geom.geometry, crop=True)
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
    print(f'Mae -- {mae} ,RMSE -- {rmse}  , R2 -- {r2}')
from shapely.geometry import mapping

def reaggregate( output_raster , merged_gdf , zone_level = 1):
    '''
    Currently, zone level is not used. But once merged_z1 has that, you can also use it. 
    '''
    print(output_raster) if not SUPRESS_VERBOSE else None 
    print(merged_gdf) if not SUPRESS_VERBOSE else None 
    with rio.open(output_raster) as raster:     
        logging.info('In reaggregate as well, we are assuming raster is single band.')
        preds = []
        trues = []
        for idx,row in merged_gdf.iterrows():

            geom = row.geometry
            geom_geojson = [mapping(geom)]
            out_image, out_transform = mask(raster, geom_geojson, crop=True)
            roi_mean = np.nanmean(out_image[out_image != src.nodata])
            val = row[ylabel+'_'+str(year)] / (row['km2_tot'] * 100)
            if(np.isnan(val)):
                continue
            preds.append(roi_mean)
            trues.append(val)

        mae = mean_absolute_error(trues, preds)
        mse = mean_squared_error(trues, preds)
        r2 = r2_score(trues, preds)
        plt.figure(idx)
        plt.scatter(x = preds , y = trues)
        plt.plot([min(preds), max(preds)], [min(trues), max(trues)], color="red", linestyle="--", label="Ideal Fit")
        metrics_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2:.2f}"
        plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, 
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
        
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Re aggregated comparision')

        plt.show()
    
    
def plots(df , xfeature , yfeature):
    sns.scatterplot(data = df,x =xfeature , y= yfeature )
    #plt.scatter(vals,model.predict(vals))
    plt.show()

chekka1 = 0
def learner(df,ylabel , model_to_use = gv.model_to_use):
   global chekka1
   global poly 
   df = df.dropna()
   X = df.drop(columns=[ylabel])
   y = df[ylabel]
   
   scaler = StandardScaler()
   #X_scaled = scaler.fit_transform(X)
   #poly = PolynomialFeatures(degree=2)
   y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
   warnings.warn("Remove y scaling in the next iteration.")
   
   chekka1 = y
   X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2)
   
   if(model_to_use == 'RFR'):
    model = RandomForestRegressor(n_estimators=2)
   elif(model_to_use == 'LR'):
    model = LinearRegression() 
   elif(model_to_use == 'SVR'):
    model = SVR()
   elif(model_to_use == 'KNR'):
    model = KNeighborsRegressor()
   else:
    raise NotImplementedError
   #model = LinearRegression()
   #model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   loss = mean_squared_error(y_test, y_pred)
   print(df) if not SUPRESS_VERBOSE else None 
   '''
   for col in X.columns:
       #plots(df , col , 'Area')
       sns.scatterplot(data = df,x =col , y= 'Area' )
       plt.show()
    '''
       
   print(f'Testing loss -- {loss}')
   return model

chekka = 0
def predictor(model , raster_path,roi,fill_value=0.0):
    global chekka
    "CHANGE LATER"
    #roi = roi.iloc[0]
    '''
    Given the trained model and raster, use the model on the raster and save the raster.
    roi is now the country's roi
    roi is now a list of country rois
    roi is now a gdf
    '''
    
    
    with rio.open(raster_path) as raster: 
        #img, transform = mask(raster, [roi], crop=True)
        img = raster.read()
        transform = raster.transform
        meta = raster.meta.copy()
        
    valid_mask = img[0] != raster.nodata 
    #fill_value = raster.nodata

    assert fill_value   == raster.nodata , f"Once predicted, make sure to use the correct nodata. Fill value -- {fill_value} raster nodata -- {raster.nodata} "
    # Assuming first band can be used for masking
    X = img[:, valid_mask].T  # Flatten valid pixels to (pixels, bands)
    predictions = model.predict(X)
    logging.warning('Inefficieny here, change it')
    if(isinstance(predictions[0], np.ndarray)):
        predictions  = np.array([val[0] for val in predictions])
    result_raster = np.full(img.shape[1:], fill_value,  dtype=np.float32)
    result_raster[valid_mask] = predictions
    
    logging.warning('LR has error here ')
    chekka = predictions
    meta.update({"dtype": "float32", "height": img.shape[1], "width": img.shape[2], "transform": transform})
    
        
    return result_raster , meta


"This function has side effects -- Dependancy with global variables. "
def get_output(gdf, ylabel, year, prediction_raster_paths , test_countries_adm0 , how = 'ml',output_path=gv.path_tiff_output ):
    '''
    Make sure country_geom is a
    
    This is a useless function for now but in the future once we have other versions/ML models, this will expand. 
    IMportant realization for this function - 
        `You are not concerned with what was used for training and what for testing anymore. 
        You have a df and an indicator path for prediction. that's about it. 
    '''
    
    fullpaths = []
    if(how == 'version1'):
        pass 
    elif(how == 'version2'):
        pass 
    elif(how == 'ml'):
        
        model = learner(gdf,ylabel = ylabel)
        output_path += '\\Raster_ML_'
        logging.info("This needs to be changed. I don't like depending on i, or that this is happening both here and in test_binder" )
        
        for i in range(len(prediction_raster_paths)): 
            indicator_path  = prediction_raster_paths[i] 
            test_country = test_countries_adm0.iloc[0]
            
            #country_geom = test_country.geometry 


            output_raster , meta = predictor(model , indicator_path ,test_country)
            logging.info("Assuming output raster always will be a single band raster.")
            fullpath = output_path+test_country.name0+'.tif'
            fullpaths.append( fullpath) 
            with rio.open(fullpath, "w", **meta) as dest:
                dest.write(output_raster, 1)
            

    else:
        raise NotImplementedError 
    return fullpaths 

    
#%%



#%%

"This function has side effects -- Dependancy with global variables. "
def test_binder( test_countries_adm0, year ,agg_variables):
    '''
    Prepare the Raster for prediction. 
    This involves creating the indicators based on the agg_variables we previously defined. 
    
    Difference in testing is that, we create separate rasters for each of the testing countries. 
    '''
    logging.info('This is currently inefficient. there is redundancy in indicator creation from training. ')
    list_of_indicator_paths = []
    for idx,grp in test_countries_adm0.groupby('name0'):
        roi = grp.geometry #check this. Make sure it is a list or a gpd
        indicators = []
        for aggvar in agg_variables:
            agg , variable = aggvar.split('_')
            indicator = get_indicator_agg(year = year , roi =roi , variable = variable ,agg=agg)  #check roi 
            indicators.append(indicator)
        output_path = gv.path_tiff_interim +'\\'+ idx+'_Raster_merged.tif'
        indicator_path = merge_raster_bands(indicators, output_path)
        list_of_indicator_paths.append(indicator_path)
        
    
    return list_of_indicator_paths 
    
        
    


def train_binder(train_merged,input_raster_path,year,train_countries_adm0,agg_variables = ['max_ndvi','mean_ndvi'],ylabel = 'Area' ):
    
    '''
    Take all the geometries of admin zones. You are not concerned with zone level anymore. 
    Return a dataframe that can be ML trained. 
    
    '''
    
    
    logging.warning("other admin zone shapefiles are still being used. ")
    #with rio.open(input_raster_path) as input_raster:
     #   assert country_adm0.crs == input_raster.crs , "CRS mismatch, reproject."
        
    assert train_countries_adm0.crs == GLOBAL_CRS , "CRS mismatch, reproject."
    
    train_countries_adm0 = train_countries_adm0.geometry

    rows = []
    logging.info('Standalone (unaggregated) variables are not implemented yet.')
    for aggvar in agg_variables:
        agg , variable = aggvar.split('_')
        single_column = []
        indicator = get_indicator_agg(year = year , roi =train_countries_adm0 , variable = variable ,agg=agg) 
        #for geom in merged['geometry']: #It is assumed that you have the respective admin zone geometry. 
        for idx,row in train_merged.iterrows():
            geom = row['geometry']
            try:
                single_column.append( gatherer(geom , input_raster_path ,indicator , variable ,year,agg_name = agg))
            # If this gives error , append empty row.
            except Exception as e:
                logging.error(f'Error in geom extract: {e}')

        rows.append(single_column)
        
    rows = [pd.DataFrame(row) for row in rows]
    df = pd.DataFrame(pd.concat(rows,axis=1))
    
    ptbl = df.copy()
    ptbl['Yield'] = train_merged['Yield_'+str(year)]
    ptbl = ptbl.dropna()
    sc = StandardScaler()
    ptbl_array = sc.fit_transform(ptbl)
    ptbl = pd.DataFrame(ptbl_array, columns=ptbl.columns)
    print(ptbl) if not SUPRESS_VERBOSE else None 
    for col in ptbl.columns:
        if(col == 'Yield'):
            continue 

        mae = mean_absolute_error(ptbl.Yield, ptbl[col])
        mse = mean_squared_error(ptbl.Yield, ptbl[col])
        r2 = r2_score(ptbl.Yield, ptbl[col])
        sns.scatterplot(data = ptbl , x =col , y = 'Yield' )
        plt.plot([min(ptbl[col]), max(ptbl[col])], [min(ptbl.Yield), max(ptbl.Yield)], color="red", linestyle="--", label="Ideal Fit")
        metrics_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2:.2f}"
        plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, 
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
        # Formatting
        plt.title(f'Prediction comparision with Yield --  {col}')
        #plt.xlabel("True Values")
        #plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.show() 
    

    #df[ylabel] = merged[str(year)]
    df[ylabel] = train_merged[ylabel+'_'+str(year)] / (train_merged['km2_tot'] * 100)
    print(df) if not SUPRESS_VERBOSE else None 
    return df 


    
    

def executer(country, crop, year ):
    
    
    '''
    
    '''
def get_agstat_with_geom(regionlist ,agstat_file, crop, zone_level =1 , shuffle = False  ):

    
    
    #Get admin 1 zones of the country. 
    countries_adm1 = get_country_zones(regionlist,zone_level=zone_level)
    assert countries_adm1.crs == GLOBAL_CRS , "CRS mismatch, reproject."
    countries_adm1['name1_SIMP'] = countries_adm1['name1'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
    
    "Agstats"
    stat = agstat.get_specific_wrap(agstat_file, crop=crop, countrylist=regionlist,kind='all',zone=zone_level)
    stat['ADM1_NAME_SIMP'] = stat['ADM1_NAME'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
    
    logging.info("The names of admin zones aren't consistent across admin_zones and agstats. I am currently merging based on all lower alphabet characters.")
    
    logging.warning("merging is being done with inner now because of spelling mismatches in agstatand GAUL")
    #assert set(countries_adm1['name1_SIMP']) == set(stat['ADM1_NAME_SIMP'])
    merged_z1 = pd.merge(stat, countries_adm1,how='inner', left_on='ADM1_NAME_SIMP', right_on='name1_SIMP')
    merged_z1 = gpd.GeoDataFrame(merged_z1[merged_z1['ADM2_NAME'].isna()])
    merged_z1['ADM_rel_NAME'] = merged_z1['name1_SIMP']
    # Ensure CRS alignment
    merged_z1 = merged_z1.to_crs(GLOBAL_CRS)
    
    if(shuffle):
        merged_z1 = merged_z1.sample(frac=1).reset_index(drop=True)
    
    
    return merged_z1
if __name__ == '__main__':
    
    agstat_file = agstat.one_big_file(crops=[],kinds=[])
    year = gv.year
    ylabel = 'Area' #Also known as kind
    agg_variables = gv.agg_variables
    train_countries = gv.train_countries
    test_countries = gv.test_countries  
    
    for crop in gv.crops:
        
        
        "Input crop mask "
        input_raster_path = gv.input_crop_map[crop]
        print(f' Input raster path  -- {input_raster_path}')
        try:
            with rio.open(input_raster_path) as src:
                GLOBAL_CRS = src.crs
        except:
            raise FileNotFoundError("Check if Crop is available.")
            
        #Get country geometry 
        train_countries_adm0 = get_country_zones(train_countries,zone_level=0,gaul=True)
        test_countries_adm0 = get_country_zones(test_countries,zone_level=0,gaul=True)
        
        merged_z1_train = get_agstat_with_geom(train_countries,agstat_file, crop,zone_level = 1)
        merged_z1_test = get_agstat_with_geom(test_countries,agstat_file, crop , zone_level = 1)
        
        
        logging.info('Test z1 df you need to treat each country seperately. Here (in main executer), Its the same but later groupby adm0_name (name0). We need to do this to get a separate tif for each testing region.')


        df = train_binder(merged_z1_train,input_raster_path =input_raster_path,year=year,ylabel=ylabel , train_countries_adm0=train_countries_adm0,agg_variables = agg_variables )
        list_of_indicator_paths = test_binder(test_countries_adm0 , year ,agg_variables)
        
        
        output_raster_paths = get_output(df ,ylabel = ylabel ,year = year , prediction_raster_paths = list_of_indicator_paths  , test_countries_adm0 = test_countries_adm0 )
        
        
        for i in range(len(output_raster_paths)):
            sample = output_raster_paths[0]

            diagnostics(sample ,input_raster_path , test_countries_adm0.iloc[i:i+1])
            reaggregate(sample , merged_z1_test , zone_level = 1)
        






        

        
        
                
        
        

