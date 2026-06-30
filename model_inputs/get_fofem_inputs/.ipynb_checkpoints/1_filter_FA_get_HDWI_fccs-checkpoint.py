 #!/usr/bin/env python
# coding: utf-8

import xarray as xr
import pandas as pd
import os
import ast
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import rasterio
from rasterio.mask import mask
from collections import Counter
import re
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path
from netCDF4 import Dataset
import wrf
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords, interplevel, xy_to_ll, ll_to_xy, destagger)


def select_fires(date_range, id_no, method):
    '''
    select_fires: Function to filter fire atlas dataset either by fire ID or by date range.
    
    In: 
    date_range: the date range to filter the fire atlas data by, must be in the format: YYYY-MM-dd, ex. 2020-09-01
    fire_id: if known, most fire IDs for the fire progressions done by Jacob correspond to NIFC fire IDs  
    method: the method to filter by, either chose date range ('dates'), dates and ID ('dates_id') or by ID ('id')
    Out: 
    Fire atlas dataset filtered by date range and/or fire id
    
    '''
    # Path to fire atlas dataset (for California)
    ROOT = Path.cwd().parents[1]
    fire_atlas_path = ROOT / "data" / "fire_atlas_CA.csv"

    # Read in fire atlas dataset, all fires from 2003-2021. At 463.3127m resolution. Also filter 
    fire_atlas = pd.read_csv(fire_atlas_path)

    # Convert doy into readable date format and add year, want to floor doy so we're rounding down
    fire_atlas['datetime'] = np.floor(fire_atlas['doy'])
    fire_atlas['datetime'] = pd.to_datetime(fire_atlas['fire_year'] * 1000 + fire_atlas['datetime'], format='%Y%j')

    if method == 'dates':  # filter fire atlas dataset based on date range alone
        filt_fire = fire_atlas.loc[((fire_atlas['datetime'] >= date_range[0]) & (fire_atlas['datetime'] <= date_range[1]))]
        
    elif method == 'id':   # filter fire atlas dataset based on fire ID alone
        filt_fire = fire_atlas.loc[((fire_atlas['fire_id'] == id_no))]
        
    elif method == 'dates_id':   # filter fire atlas dataset based on fire ID AND date range
        filt_fire = fire_atlas.loc[((fire_atlas['fire_id'] == id_no))]
        filt_fire = filt_fire.loc[((filt_fire['datetime'] >= date_range[0]) & (filt_fire['datetime'] <= date_range[1]))]
        
    # Assign index for debugging/easier processing of the outputs    
    filt_fire['og_index'] = range(1, len(filt_fire) + 1) 

    ### Get fuel classes
    fccs = get_fccs(filt_fire)
    fccs = fccs.sort_values(by=['datetime'])

    # Group by datetime, get all unique fuel classes present in all fire atlas grid cells for a day, sum the totals for each fuel class
    fccs_agg = fccs.groupby(['datetime','fire_id']).agg({
        'num_fccs_cells': 'sum',
        'FCCS_IDs': lambda x: dict(sum((Counter(d) for d in x), Counter()))
    }).reset_index()

    fccs = fccs.drop(columns=['FCCS_IDs','num_fccs_cells'])
    fccs_agg = pd.merge(fccs, fccs_agg, on=['datetime','fire_id'])
    
    # Calculate growth for each fire if there are multiple fire IDs
    def calc_daily_growth(fire): 
        # Calculate wildfire growth each day from daily perimeters
        fire['daily_growth'] = fire.groupby(fire['datetime'].dt.date)['datetime'].transform('count')
        fire['daily_growth'] = (fire['daily_growth'] * 463.3127 * 463.3127)/ 4046.8564   # grid cells are 463.3127m x 463.3127m 
        fire = fire.loc[fire.index.repeat(24)].reset_index(drop=True)
        fire["datetime"] = pd.to_datetime(fire["datetime"])
        fire["datetime"] += pd.to_timedelta(fire.groupby(fire.index // 24).cumcount(), unit="h")

        return fire

    fire_growth = fccs_agg.groupby('fire_id').apply(calc_daily_growth).reset_index(drop=True)
    
    return fire_growth

def get_fccs(dataset):
    ''' 
    get_fccs: Function to get FCCS class counts for each 463.3127 m x 463.3127 m "grid cell", point data is buffered/a box is added around each
    input point/centroid. Raster masks are used to loop through each box/geometery in the created fire atlas geodataframe and count FCCS classes. Note 
    the resolution of the FCCS data is 30 m.
    
    In: 
    dataset: Filtered dataset by date range/fire ID 
    Out: 
    Dataset with added FCCS class counts
    
    '''
    # Convert dataset to geodataframe
    dataset = gpd.GeoDataFrame(
        dataset,
        geometry=[Point(lon, lat) for lon, lat in zip(dataset['lon'], dataset['lat'])],
        crs="EPSG:4326"
    )
    
    # Open the TIF file containing FCCS classes for all of California
    DATA_PATH = "path/to/downloaded_dataset" # Data is publicly available and can be downloaded from the source referenced in the associated publication.
    fccs_tif = DATA_PATH

    with rasterio.open(fccs_tif) as src:
        tif_crs = src.crs
    
    # Reproject fire atlas points to CRS of the raster (ensure its a metric CRS in units of meters)
    fa_gdf = dataset.to_crs(tif_crs)

    radius = 463.3127 / 2  # Half the size of the grid dimensions 463.3127x463.3127, extend this distance
    # out from centroid
    fa_gdf['geometry'] = fa_gdf['geometry'].apply(
        lambda point: box(
            point.x - radius,  # Min X
            point.y - radius,  # Min Y
            point.x + radius,  # Max X
            point.y + radius   # Max Y
        )
    )
    
    # Open the TIF file containing the FCCS classes, will be masking each grid cell/polygon and counting
    # the FCCS classes present in each grid cell
    with rasterio.open(fccs_tif) as src:
        tif_crs = src.crs
        # Get land class counts per grid cell using zonal statistics
        counts_list = []
        
        for _, row in fa_gdf.iterrows():
            # Mask raster with the current grid cell/fire atlas polygon (row.geometry)
            out_image, out_transform = mask(src, [row.geometry], crop=True, nodata=0)
        
            # Flatten the array and count unique values
            unique, counts = np.unique(out_image[out_image != 0], return_counts=True)
            counts_dict = dict(zip(unique, counts))
        
            # Append to the results list
            counts_list.append(counts_dict)
    
    # Add the FCCS class counts to the GeoDataFrame, also reproject to get coordinates in degrees
    fa_gdf["FCCS"] = counts_list
    fa_gdf = fa_gdf.to_crs(epsg=4326) 
    
    # Convert GeoDataFrame to pandas dataframe
    fccs_counts = pd.DataFrame(fa_gdf)

    # Get the main fuel class, separate from sub groups of fuel classes, ex. 2140112 & 2140322 are class 214
    # When combining the sub groups add the total number of each in the main fuel class/group
    fccs_counts['FCCS_IDs'] = fccs_counts['FCCS'].apply(
        lambda d: {
            re.sub(r'0.*', '', str(k)): sum(v for k_, v in d.items() if re.sub(r'0.*', '', str(k_)) == re.sub(r'0.*', '', str(k)))
            for k, v in d.items()
        }
    )
    fccs_counts['num_fccs_cells'] = fccs_counts['FCCS_IDs'].apply(lambda d: sum(d.values()))

    return fccs_counts


def get_hourly_hdwi_reanalysis(group):
    '''
    get_hourly_hdwi_reanalysis: Function to get HDWI for each time at each point using the WRF reanalysis product
    
    In: 
    group: Groups used for parallelization, created from grouping previously filtered Fire Atlas dataset by datetime
   
    Out: 
    Groups with calculated HDWIs
    
    '''
    group['datetime'] = pd.to_datetime(group['datetime'])
    year = group['datetime'].iloc[0].year
    month = group['datetime'].iloc[0].month
    day = group['datetime'].iloc[0].day
    hour = group['datetime'].iloc[0].hour

    # Base path where each year of WRF outputs are located
    wrfout_basepath = '/path/to/surface/meteorology'
    # Construct the file path so we're getting the wrfout file
    wrfout_file = f"{str(year)}/{month:02d}/wrfsfc_d03_{year:04d}{month:02d}{day:02d}_{hour:02d}00.nc"
    wrfout_filepath = wrfout_basepath + wrfout_file

    if not os.path.exists(wrfout_filepath):
        print(f"Skipping {wrfout_filepath}, file does not exist.")
        return None
        
    # Open WRF output file
    ncfile = Dataset(wrfout_filepath)

    # Calculate HDWI 
    # Extract the necessary variables needed to calculate HDWI
    t2 = getvar(ncfile, "T2", timeidx=wrf.ALL_TIMES)
    q2 = getvar(ncfile, "Q2", timeidx=wrf.ALL_TIMES)
    u = getvar(ncfile, "U10", timeidx=wrf.ALL_TIMES)
    v = getvar(ncfile, "V10", timeidx=wrf.ALL_TIMES)
    psfc = getvar(ncfile, "PSFC", timeidx=wrf.ALL_TIMES)
    ws = np.sqrt(u**2 + v**2)
    wd = (270 - np.arctan2(v, u) * (180 / np.pi)) % 360

    # Saturation vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * (t2 - 273.15)) / (t2 - 29.65))
    
    # Vapor pressure (hPa)
    # e = q * p / (0.622 + 0.378 * q)
    e = (q2 * psfc) / (0.622 + 0.378 * q2)
    e = e / 100.0   # convert from Pa to hPa
    
    # Vapor pressure deficit (VPD)
    vpd = es - e
    
    # Calculate HDW index, (wind speed * vapor pressure deficit)
    hdw = ws * vpd

    # Get lat/lons from fire atlas data, convert to numpy arrays
    fa_lats = group['lat'].to_numpy()
    fa_lons = group['lon'].to_numpy()

    # map fire atlas lat/lons to wrf i,j indices
    ny, nx = hdw.shape
    indices = np.array([ll_to_xy(ncfile, lat, lon) for lat, lon in zip(fa_lats, fa_lons)])
    indices = np.round(indices).astype(int)

    flat_indices = indices[:, 1] * nx + indices[:, 0]

    # Extract HDW values correctly
    hdw_values = np.take(hdw.values.flatten(), flat_indices)
    vpd_values = np.take(vpd.values.flatten(), flat_indices)
    es_values = np.take(es.values.flatten(), flat_indices)
    e_values = np.take(e.values.flatten(), flat_indices)
    t2_values = np.take(t2.values.flatten(), flat_indices)
    ws_values = np.take(ws.values.flatten(), flat_indices)
    wd_values = np.take(wd.values.flatten(), flat_indices)

    group_met = pd.DataFrame({
            "lon": fa_lons,
            "lat": fa_lats,
            "doy": group['doy'].to_numpy(),
            "datetime": group['datetime'],
            "fire_id": group['fire_id'],
            "daily_growth": group['daily_growth'],
            "vpd": vpd_values,
            "es": es_values,
            "e": e_values,
            "t2": t2_values,
            "ws": ws_values,
            "wd": wd_values,
            "HDWI": hdw_values,
            "og_index": group['og_index'],
            "num_fccs_cells": group['num_fccs_cells'],
            "FCCS_IDs": group['FCCS_IDs']
    })
    return group_met

### Create groups from filtered dataset for parallelization
MAX_PROCESSES = min(8, multiprocessing.cpu_count())
def group_fa(dataset):
    # Create groups based on datetime, so WRF files can be opened once for each group
    grouped = dataset.groupby('datetime')
    
    # Use multiprocessing for parallel processing
    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        results = pool.map(get_hourly_hdwi_reanalysis, [group for _, group in grouped])
    
    results = [result for result in results if result is not None]
    
    # Concatenate the results 
    group_hdwi = pd.concat(results, ignore_index=True)
    
    return group_hdwi

### Filter Fire Atlas dataset by fire ID:
creekfire = '2020-CASNF-001391'
lakefire = '2020-CAANF-003273'
applefire = '2020-CARRU-096640'
claremont = '2020-CAPNF-001302'
walker = '2019-CAPNF-001324'
scu_complex = '2020-CASCU-005740'
bobcat = '2020-CAANF-003687'
caynp = '2020-CAYNP-000054'

# specify: 'id', 'dates', 'dates_id'
# method = 'dates'
method = 'id'
dates = ['2020-01-01', '2021-01-01']

### Call functions, initiate parallelization   
if __name__ == "__main__":
    filt_fire = select_fires(0, creekfire, method)
    
    # Process the dataset using multiprocessing
    hdwis = group_fa(filt_fire)

    # Update to a preferred output directory
    # hdwis.to_csv('path/to/output/MP_2020_creekfire_HDWI_reanalysis.csv')

    
