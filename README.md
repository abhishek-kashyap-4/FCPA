
## FCPA
Start - Dec 4, 2025	

End - Current 


Started development for Fractional Crop Percentage Area in Ukraine, Russia, China for spring and winter wheat. 


Data gathered for this work includes -  
1. Agricultural statistics at Admin 1 level across the years 
2. Input raster crop  mask (global)
3. Vector datasets on Admin 1,2,3,4 levels for Ukraine, Russia (so far) 
Methodology: 
	Method is divided into units - each unit works for a single country, single crop-type. Geometry is filtered according to the vector datasets. For initial version, rasters are created which have the same properties of input raster, but have the percentage yield values existing in agriculture stats. 

(N) -  the ag stat data is given such that the crops are different files and the countries are in the same file. However, it makes more sense for me that the country would be split first and then crop. In the future, we will be working on more countries and crops so we need to make a decision right now. I think it is better to merge everything into a single file, and select only the ones that are useful to us. 

It is possible that data doesn’t exist for the specific croptype /country etc combination. Deal with this case. 

