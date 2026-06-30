#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import multiprocessing as mp
import os

from datetime import datetime, timedelta
import numpy as np
import csv
import os
import pandas as pd
from pathlib import Path

def fix_datetime_formatting(x):
    x = str(x).strip()
    if len(x) == 10:  # Format 'YYYY-MM-DD'
        return x + ' 00:00:00'
    return x
    
# Function to process a chunk of data (a complete group)
def agg_fofout(group):
    
    # Since crown heat fluxes represent 60 seconds of burning, set the crown heat flux to only the first 4 time steps
    crown_fi_60s = group['crown_fi'].iloc[0]
    group['crown_fi'] = 0
    group['crown_fi'] = group['crown_fi'].astype(float)
    group.loc[group.index[:4], 'crown_fi'] = crown_fi_60s

    # Convert 'date_time' to datetime objects
    group['orig_time'] = group['date_time']
    
    # Calculate delta t (each interval is 15 seconds in timeidx) and apply it to 'date_time'
    group['date_time'] = pd.to_datetime(group['date_time'], format='%Y%m%d%H')
    # # Add delta t (or the time a grid cell burned) to the original input time to create a "new" time
    group['new_datetime'] = group.apply(lambda row: row['date_time'] + timedelta(seconds=row['timeidx'] * 15), axis=1)
    
    # Reformat "new" datetime and set as the index
    group['Time'] = group['new_datetime'].dt.strftime('%Y-%m-%d %H:%M:%s')

    # After calculating the "new" times, pad the missing time stamps for each hour (since some things burn under an hour or into consecutive hours)
    # Identify the last timestamp for each group
    last_time = group['new_datetime'].max()
    
    # Compute the end of that hour
    end_of_hour = last_time.replace(minute=59, second=45)
    
    # Generate all expected timestamps in 15s intervals from last_time+15s to the end of the hour
    new_times = pd.date_range(
        start=last_time + timedelta(seconds=15),
        end=end_of_hour,
        freq='15s'
    )
    if len(new_times) > 0:
        # Create the new rows
        new_rows = pd.DataFrame({'new_datetime': new_times})
    
        # Set individual column values, heat fluxes and emissions to 0
        new_rows['fuelbed_name'] = group['fuelbed_name'].iloc[0]
        new_rows['lat'] = group['lat'].iloc[0]
        new_rows['lon'] = group['lon'].iloc[0]
        new_rows['date_time'] = group['date_time'].iloc[0]
        new_rows['timestep'] = group['timestep'].iloc[0]
        new_rows['fmois1h'] = group['fmois1h'].iloc[0]
        new_rows['fmois10h'] = group['fmois10h'].iloc[0]
        new_rows['fmois100h'] = group['fmois100h'].iloc[0]
        new_rows['fmois1kh'] = group['fmois1kh'].iloc[0]
        new_rows['fmoislit'] = group['fmoislit'].iloc[0]
        new_rows['fmoisduff'] = group['fmoisduff'].iloc[0]
        new_rows['percent_crown_burn'] = group['percent_crown_burn'].iloc[0]
        new_rows['fuelcat'] = group['fuelcat'].iloc[0]
        new_rows['dufftype'] = group['dufftype'].iloc[0]
        new_rows['met_dat_path'] = group['met_dat_path'].iloc[0]
        new_rows['ambient_temp'] = group['ambient_temp'].iloc[0]
        new_rows['windspeed'] = group['windspeed'].iloc[0]
        new_rows['emission_factors'] = group['emission_factors'].iloc[0]
        new_rows['fire_type'] = group['fire_type'].iloc[0]
        new_rows['consumed_flaming'] = group['consumed_flaming'].iloc[0]
        new_rows['consumed_smoldering'] = group['consumed_smoldering'].iloc[0]
        new_rows['fi'] = 0
        new_rows['crown_fi'] = 0
        new_rows['pm25'] = 0
        new_rows['pm10'] = 0
        new_rows['co2'] = 0
        new_rows['con_ob'] = group['con_ob'].iloc[0]
        new_rows['fire_id'] = group['fire_id'].iloc[0]
        new_rows['hourly_growth'] = group['hourly_growth'].iloc[0]
        new_rows['FCCS_class_percent'] = group['FCCS_class_percent'].iloc[0]
        new_rows['group'] = group['group'].iloc[0]
       
        # Continue timeidx from last known index
        last_timeidx = group['timeidx'].max()
        new_rows['timeidx'] = range(last_timeidx + 1, last_timeidx + 1 + len(new_rows))
    
        # Append new rows
        group = pd.concat([group, new_rows], ignore_index=True)
       
    group = group.sort_values('new_datetime').reset_index(drop=True)
    group.set_index('new_datetime', inplace=True)
    
    # Aggregate the fofem outputs hourly, for the inputs that are constant for each time step take the first instance,
    # for other variables either sum or take the mean depending on what results you're looking for
    aggregated = group.resample('h').agg({
        'fuelbed_name': 'first',
        'lat' : 'first',
        'lon' : 'first',
        'date_time': 'first',
        'timestep': 'first',
        'fmois1h': 'first',
        'fmois10h': 'first',
        'fmois100h': 'first',
        'fmois1kh': 'first',
        'fmoislit': 'first',
        'fmoisduff': 'first',
        'percent_crown_burn': 'first', 
        'fuelcat': 'first', 
        'dufftype': 'first',
        'met_dat_path': 'first',
        'ambient_temp': 'first',
        'windspeed': 'first', 
        'emission_factors': 'first', 
        'fire_type': 'first',
        'consumed_flaming': 'first',
        'consumed_smoldering': 'first', 
        'pm25':'mean',
        'pm10':'mean',
        'co2':'mean',
        'fi': 'mean',
        'crown_fi':'mean',
        'con_ob': 'first',
        'fire_id': 'first', 
        'hourly_growth': 'first',
        'FCCS_class_percent': 'first',
        'group': 'first',
    }).reset_index()
    
   
    return aggregated


# Function to read the CSV and chunk by group
def chunk_by_group(df):
    
    # Identify groups based on timeidx == 0, as 0 indicates the start of a new group/unique fofem input
    df['timeidx_zero'] = df['timeidx'] == 0

    df['group'] = df['timeidx_zero'].cumsum()  # Assign a group number to each set of rows

    # Split DataFrame into chunks based on group number, this is for multiprocessing
    
    group_chunks = [group for _, group in df.groupby(['group'])]
    
    return group_chunks


# Main function to process the CSV and handle the parallel processing, most of this is standard multiprocessing code with a few modifications
def process_outputs(df, output_file):

    # Step 1: Chunk the data by group
    group_chunks = chunk_by_group(df)

    # Step 2: Set up multiprocessing
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(agg_fofout, (chunk,)) for chunk in group_chunks]

    pool.close()
    pool.join()

    if output_file != 0:
         # Step 3: Collect the results and write them to the output CSV
        with open(output_file, 'w') as f_out:
            header_written = False
            for result in results:
                processed_data = result.get()
    
                if not header_written:
                    processed_data.to_csv(f_out, header=True, index=False)
                    header_written = True
                else:
                    processed_data.to_csv(f_out, header=False, index=False)
                    
    elif output_file == 0:
        # Collect all processed DataFrames
        all_dfs = [result.get() for result in results]
        
        # Concatenate them into a single DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)

        return combined_df


### Function to calculate weighted heat fluxes
def calc_mean_heatflux(fofem_outputs):
    
    ### Updated mean heat fluxes calculation
    fofem_outputs['weighted_fi'] = fofem_outputs['fi'] * fofem_outputs['FCCS_class_percent']
    
    # Sum together the weighted heat fluxes by the hour using the "new" datetimes
    fofem_outputs['new_datetime'] = pd.to_datetime(fofem_outputs['new_datetime'])
    fofem_outputs.set_index('new_datetime', inplace=True)
    weighted_hf = fofem_outputs.resample('h').agg({
        'date_time':'first',
        'lat':'first',
        'lon':'first',
        'fire_id':'first',
        'FCCS_class_percent':'sum',
        'weighted_fi': 'sum',
        # 'fmois1kh':'first',
        # 'fmois100h':'first',
        # 'fmois10h':'first',
        # 'fmois1h':'first',
        # 'ambient_temp':'first',
        # 'windspeed':'first',
        'hourly_growth':'first'
    })
    
    # Need to renormalize by dividing by the sum of the weights, which in some cases is > 1
    weighted_hf['weighted_fi'] = weighted_hf['weighted_fi'] / weighted_hf['FCCS_class_percent']
    
    # Format weighted heat flux dataframe
    weighted_hf = weighted_hf.reset_index()

    weighted_hf.columns = ['new_datetime','date_time','lat','lon','fire_id','FCCS_class_percent','weighted_fi','hourly_growth']
    weighted_hf['date_time'] = pd.to_datetime(weighted_hf['date_time'])

    return weighted_hf

### Function to calculate active burn area
def calc_rel_burnarea(fofem_outputs):
    
    ### Multiply hourly growth by fuel class weightings to approx growth associated with each fuel class
    fofem_outputs['growth_byclass'] = fofem_outputs['hourly_growth'] * fofem_outputs['FCCS_class_percent']
    
    # Sum together the weighted growth by the hour using the "new" datetimes to estimate relative burn area (accounting for active burn area/rollover)
    fofem_outputs['new_datetime'] = pd.to_datetime(fofem_outputs['new_datetime'])
    fofem_outputs.set_index('new_datetime', inplace=True)
    rel_burnarea = fofem_outputs.resample('h').agg({
    'fire_id':'first',
    'growth_byclass': 'sum'
})
    
    # Format weighted burn area dataframe
    rel_burnarea = rel_burnarea.reset_index()
    rel_burnarea.columns = ['date_time','fire_id','rel_burnarea']
    rel_burnarea['date_time'] = pd.to_datetime(rel_burnarea['date_time'])

    return rel_burnarea



