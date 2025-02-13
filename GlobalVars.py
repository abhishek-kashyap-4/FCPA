



#PATHS
path_agstat = 'Data\\agstats'
path_admin_zones = 'Data\\Admin_Zones'
path_tiff_input = 'Data\\Input'
path_tiff_output = 'Data\\Output\\tiffs'
path_tiff_interim = 'Data\\Interim\\tiffs'


year = 2015
model_to_use = 'RFR'
models_avail = ['RFR' , 'LR' , 'SVR' , 'KNR']
#countries = ['Ukraine' , 'Russia']
years = list(range(2001,2024))
years = list(range(2020,2023))
crops = ['WinterWheat']

input_crop_map = {
                        'SpringWheat': 'Data\crop_masks\Percent_Spring_Wheat.tif' ,
                        'WinterWheat': 'Data\crop_masks\Percent_Winter_Wheat.tif' ,
                        }


train_countries = ['Poland','Ukraine']
test_countries = ['Ukraine']


appendages = {
    'hdx_to_gaul' : { 
        'Russian Federation' : 'Russia' 
        
        }
    
    
    }