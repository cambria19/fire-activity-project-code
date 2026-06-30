#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import math
import datetime as dt
from scipy import stats
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from math import sin, cos, sqrt, atan2, radians

import sys
import import_ipynb
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
import xarray as xr
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, box
import rasterio
from pathlib import Path
from rasterio.mask import mask
import ast
import re
import matplotlib.patches as mpatches
import numpy as np
from netCDF4 import Dataset
import wrf
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords, interplevel, xy_to_ll, ll_to_xy, destagger)



def get_fuelmois(dataset):
    ''' 
    get_fuelmois: Function to get average 1hr, 10hr, 100hr, and 1000hr fuel moisture at each point for each day.
    
    In: 
    Input dataset that is filtered by date range and/or fire ID
    Out: 
    Dataset with added fuel moisture of each class
    
    '''
    dataset = dataset.reset_index(drop=True) 
    
    # Base path for fuel moisture netCDFs (all months/years) and fuel moisture coordinates
    DATA_PATH = "path/to/downloaded_dataset"  # Data is publicly available and can be downloaded from the source referenced in the associated publication.
    
    fmrp_basepath = f"{DATA_PATH}/fmrp_"
    fmrp_coords_filepath = f"{DATA_PATH}/fmrp_coordinates.nc" # Data is publicly available and can be downloaded from the source referenced in the associated publication.

    # Open fuel moisture coordinates netCDF
    # Open moisture coordinates netCDF
    sep_fmrp_coords = xr.open_dataset(fmrp_coords_filepath)
    fmrp_lat = sep_fmrp_coords['XLAT'][:, 0].values
    fmrp_lon = sep_fmrp_coords['XLONG'][0, :].values

    # Build k-d tree for nearest neighbor lookup
    lat_lon_pairs = np.column_stack([fmrp_lat.repeat(len(fmrp_lon)), np.tile(fmrp_lon, len(fmrp_lat))])
    tree = cKDTree(lat_lon_pairs)

    # Extract lat/lon from dataset and find nearest indices
    query_points = dataset[['lat', 'lon']].values
    _, nearest_idxs = tree.query(query_points)
    lat_indices, lon_indices = np.divmod(nearest_idxs, len(fmrp_lon))

    # Add empty fuel moisture columns
    fuelmois_vars = ["1h_FM", "10h_FM", "100h_FM", "1000h_FM"]
    for col in fuelmois_vars:
        dataset[col] = np.nan

    # Process by month/year which is how netCDFs are organized
    dataset['year'] = dataset['datetime'].dt.year
    dataset['month'] = dataset['datetime'].dt.month
    grouped = dataset.groupby(['year', 'month'])

    results = []
    
    for (year, month), group in grouped:
        # Expand lat/lon indices to match new dataset length
        group_indices = group.index.to_numpy()
        lat_indices_filt = lat_indices[group_indices]
        lon_indices_filt = lon_indices[group_indices]
        
        month_str = str(month).zfill(2)
        fmrp_filepath = f"{fmrp_basepath}{year}_{month_str}.nc"
        print(fmrp_filepath)

        # Skip if file not found, i.e. years 2021 and later
        if not os.path.exists(fmrp_filepath):
            print(f"File not found, skipping: {fmrp_filepath}")
            continue
        
        with xr.open_dataset(fmrp_filepath) as fmrp_data:
            fmrp_data = fmrp_data.set_coords('Times')
            
            # Extract variables
            onehr_fm = fmrp_data['1h_FM'].resample(Times='1D').mean()
            tenhr_fm = fmrp_data['10h_FM'].resample(Times='1D').mean()
            hunhr_fm = fmrp_data['100h_FM'].resample(Times='1D').mean()
            thohr_fm = fmrp_data['1000h_FM'].resample(Times='1D').mean()

            # Get the day indices
            day_indices = group["datetime"].dt.day.values - 1

            # Extract fuel moistures
            fm_1hr = onehr_fm.isel(
            Times=xr.DataArray(day_indices, dims="points"),
            south_north=xr.DataArray(lat_indices_filt, dims="points"),
            west_east=xr.DataArray(lon_indices_filt, dims="points")
        )
            fm_10hr = tenhr_fm.isel(
            Times=xr.DataArray(day_indices, dims="points"),
            south_north=xr.DataArray(lat_indices_filt, dims="points"),
            west_east=xr.DataArray(lon_indices_filt, dims="points")
        )
            fm_100hr = hunhr_fm.isel(
            Times=xr.DataArray(day_indices, dims="points"),
            south_north=xr.DataArray(lat_indices_filt, dims="points"),
            west_east=xr.DataArray(lon_indices_filt, dims="points")
        )
            fm_1000hr = thohr_fm.isel(
            Times=xr.DataArray(day_indices, dims="points"),
            south_north=xr.DataArray(lat_indices_filt, dims="points"),
            west_east=xr.DataArray(lon_indices_filt, dims="points")
        )
        
            # Assign to dataframe columns
            group["1h_FM"] = fm_1hr.values
            group["10h_FM"] = fm_10hr.values
            group["100h_FM"] = fm_100hr.values
            group["1000h_FM"] = fm_1000hr.values

            results.append(group)

            # Check to see if results is empty, meaning there are missing inputs
            if not results:
                raise ValueError("Time frame not valid/input fuel moisture data missing. Exiting get_ros_inputs function.")
    return pd.concat(results, ignore_index=True)


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
    fccs_counts['FCCS_ID'] = fccs_counts['FCCS'].apply(
        lambda d: {
            re.sub(r'0.*', '', str(k)): sum(v for k_, v in d.items() if re.sub(r'0.*', '', str(k_)) == re.sub(r'0.*', '', str(k)))
            for k, v in d.items()
        }
    )
    
    fccs_counts['FCCS_ID'] = fccs_counts['FCCS_ID'].apply(lambda d: int(max(d, key=d.get)) if d else 0)
    return fccs_counts


def get_hourly_fuelmois(dataset):
    ''' 
    get_fuelmois: Function to get the 1hr, 10hr, 100hr, and 1000hr fuel moisture at each point for each hour.
    
    In: 
    Input dataset that is filtered by date range and/or fire ID
    Out: 
    Dataset with added fuel moisture of each class
    
    '''
    # Base path for fuel moisture netCDFs (all months/years) and fuel moisture coordinates
    DATA_PATH = "path/to/downloaded_dataset"  # Data is publicly available and can be downloaded from the source referenced in the associated publication.
    
    fmrp_basepath = f"{DATA_PATH}/fmrp_"
    fmrp_coords_filepath = f"{DATA_PATH}/fmrp_coordinates.nc" # Data is publicly available and can be downloaded from the source referenced in the associated publication.
    
    # Open fuel moisture coordinates netCDF
    # Open moisture coordinates netCDF
    sep_fmrp_coords = xr.open_dataset(fmrp_coords_filepath)
    fmrp_lat = sep_fmrp_coords['XLAT'][:, 0].values
    fmrp_lon = sep_fmrp_coords['XLONG'][0, :].values

    # Build k-d tree for nearest neighbor lookup
    lat_lon_pairs = np.column_stack([fmrp_lat.repeat(len(fmrp_lon)), np.tile(fmrp_lon, len(fmrp_lat))])
    tree = cKDTree(lat_lon_pairs)

    # Extract lat/lon from dataset and find nearest indices
    query_points = dataset[['lat', 'lon']].values
    _, nearest_idxs = tree.query(query_points)
    lat_indices, lon_indices = np.divmod(nearest_idxs, len(fmrp_lon))

    # Add empty fuel moisture columns
    fuelmois_vars = ["1h_FM", "10h_FM", "100h_FM", "1000h_FM"]
    for col in fuelmois_vars:
        dataset[col] = np.nan

    # Process by month/year which is how netCDFs are organized
    dataset['year'] = dataset['datetime'].dt.year
    dataset['month'] = dataset['datetime'].dt.month
    grouped = dataset.groupby(['year', 'month'])

    results = []
    
    for (year, month), group in grouped:
        # Expand lat/lon indices to match new dataset length
        group_indices = group.index.to_numpy()
        lat_indices_filt = lat_indices[group_indices]
        lon_indices_filt = lon_indices[group_indices]
        
        month_str = str(month).zfill(2)
        fmrp_filepath = f"{fmrp_basepath}{year}_{month_str}.nc"
        print('Working on:', fmrp_filepath)

        with xr.open_dataset(fmrp_filepath) as fmrp_data:
            # Extract variables
            onehr_fm = fmrp_data["1h_FM"].values
            tenhr_fm = fmrp_data["10h_FM"].values
            hunhr_fm = fmrp_data["100h_FM"].values
            thohr_fm = fmrp_data["1000h_FM"].values
            
            # Get the hour index
            days = group["datetime"].dt.day.values
            hours = group["datetime"].dt.hour.values
            hour_indices = (days * 24 - 25) + hours
   
            # Extract fuel moistures
            fm_1hr = onehr_fm[hour_indices, lat_indices_filt, lon_indices_filt]
            fm_10hr = tenhr_fm[hour_indices, lat_indices_filt, lon_indices_filt]
            fm_100hr = hunhr_fm[hour_indices, lat_indices_filt, lon_indices_filt]
            fm_1000hr = thohr_fm[hour_indices, lat_indices_filt, lon_indices_filt]
            
            # Assign to dataframe columns
            group.loc[:, "fmois1h"] = fm_1hr
            group.loc[:, "fmois10h"] = fm_10hr
            group.loc[:, "fmois100h"] = fm_100hr
            group.loc[:, "fmois1kh"] = fm_1000hr
            
            results.append(group)

    return pd.concat(results, ignore_index=True)


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
    # print(filt_fire)
    
    ### Get fuel classes, elevation, and variance in slope
    elev = get_elev(filt_fire)
    slp = get_slp_var(elev)

    return slp

