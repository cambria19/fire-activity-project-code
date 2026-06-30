#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.optimize import root_scalar

### Updated (4/4) modified HDWI, based on fraction of daily growth relative to cumulative total
def relative_frac(group):

    # Find hourly maximum HDWI and total growth each day (previously calculated)
    hourly_hdwi = group.groupby('datetime', as_index=False).agg({
        'lat': 'first',
        'lon': 'first',
        'fire_id': 'first',
        'HDWI': 'max',
        'ambient_temp': 'mean',
        'windspeed': 'mean',
        'num_fccs_cells': 'first',
        'FCCS_IDs':'first',
        'daily_acres_fireatlas': 'first'
    })

    hourly_hdwi['date'] = hourly_hdwi['datetime'].dt.date

    # Get daily growth
    daily_growth = hourly_hdwi.groupby(hourly_hdwi['datetime'].dt.date).agg({
        'date': 'first',
        'fire_id': 'first',
        'daily_acres_fireatlas': 'first'
    }).reset_index(drop=True)

    # Calculate the fraction of each day's growth relative to cumulative growth
    part_fa = 'daily_acres_fireatlas'
    daily_growth['relative_frac'] = daily_growth[part_fa]/ daily_growth[part_fa].cumsum()
    
    # Identify the peak in HDWI for the day, will represent the daily peak in fire growth 
    max_hdwi_indices = hourly_hdwi.groupby(hourly_hdwi['datetime'].dt.date)['HDWI'].idxmax()

    # Map the max timestamps back to the dataframe by day
    hourly_hdwi['max_hdwi_index'] = hourly_hdwi['datetime'].dt.date.map(lambda date: hourly_hdwi.loc[max_hdwi_indices[date], 'datetime'].hour)
    hourly_hdwi = pd.merge(hourly_hdwi, daily_growth, on='date')
    hourly_hdwi = hourly_hdwi.drop(columns=[col for col in hourly_hdwi.columns if col.endswith('_x')])
    hourly_hdwi = hourly_hdwi.rename(columns=lambda x: x.rstrip('_y'))
    
    def calc_modified_hdwi(group_hdwi):
        # Check if we have full days of data (24-hour periods)
        if len(np.unique(group_hdwi['datetime'].dt.hour)) != 24:
            adjusted_hdwi = np.array(group_hdwi['HDWI'])
            peak_index = 'invalid'   # Not assigning a peak value in HDWI because we don't have the full 24 hours

        elif np.unique(group_hdwi['relative_frac']) == 1.0:   # if relative_frac = 1.0, this is the first day of the fire
            group_hdwi = group_hdwi[21:]
            adjusted_hdwi = np.array(group_hdwi['HDWI'])
            peak_index = 'invalid'
            
        else:
            adjusted_hdwi = np.array(group_hdwi['HDWI'])
            peak_index = group_hdwi['max_hdwi_index'].iloc[0]  # Peak hour index 
            
        smooth_factor = 0.5 - (group_hdwi['relative_frac'].iloc[0] * 0.1)  # adjust based on growth fraction
        peak_multiplier = group_hdwi['relative_frac'].iloc[0] * 10  # increase/decrease peak
                
        # Gradually scale values before the peak hour
        if peak_index == 'invalid': 
            print('Skipping HDWI profile modification')
            
        else:
            
            for i in range(peak_index + 1):
                og_hdwi = np.array(group_hdwi['HDWI'])
    
                if peak_index > 0:
                    peak_scaling = 1 + (peak_multiplier - 1) * (i / peak_index) - smooth_factor
                else:
                    peak_scaling = 0.5 
                    
                # Adjust HDWI values before the peak
                adjusted_hdwi[i] = peak_scaling * og_hdwi[i]
            
            # Gradually scale values after the peak hour (exponential smoothing)
            for i in range(peak_index + 1, 24):  # Modify after the peak hour
                adjusted_hdwi[i] = (1 - smooth_factor) * adjusted_hdwi[i - 1] + smooth_factor * og_hdwi[i]
                
            # After the peak hour (for times that wrap around after midnight)
            for i in range(0, peak_index):  # Modify before the peak hour, considering continuation into next 24 hrs
                adjusted_hdwi[i] = (1 - smooth_factor) * adjusted_hdwi[i - 1] + smooth_factor * og_hdwi[i]
                
        # Remove any negative values
        adjusted_hdwi = np.abs(adjusted_hdwi) 
        group_hdwi['hourly_growth'] = (adjusted_hdwi / np.sum(adjusted_hdwi)) * group_hdwi[part_fa]   
        group_hdwi['sum_hourly_growth'] = np.sum(group_hdwi['hourly_growth'])

        return group_hdwi
                
    modified_hdwi = hourly_hdwi.groupby(hourly_hdwi['datetime'].dt.date).apply(calc_modified_hdwi).reset_index(drop=True)
    
    ### Assign values for first day growth that follow an exponential curve
    firstday_growth = modified_hdwi['daily_acres_fireatlas'][0]
    
    # In case a fire is only one day, assign next growth as equal to the first day of growth
    if len(modified_hdwi) == 1 or len(modified_hdwi) == 2:
        return modified_hdwi
        
    elif len(modified_hdwi) == 3:
        next_growth = firstday_growth
        
    else:
        next_growth = modified_hdwi['hourly_growth'][3]
    
    ### Use a root-finding method to approximate exponential growth on the first day of growth
    
    # Function whose root we want to find
    def find_root(x):
        base = np.exp(x)
        values = np.logspace(0.01, np.log(next_growth), 3, base=base)
        return np.sum(values) - firstday_growth
    try:
        solution = root_scalar(find_root, bracket=[-10, 10], method='brentq')
        
        if solution.converged:
            # print(f"Solution found: x = {solution.root}")
            # Assign these to the last 3 hours of the 1st day 
            modified_hdwi['hourly_growth'][0:3] = np.logspace(0.01, np.log(next_growth), 3, base=np.exp(solution.root))
        else:
            print("Root finding did not converge.")

        
    except ValueError as e:
        print(f"Root solving failed: {e}. Using default exponential growth.")
        r = (-1 + np.sqrt(1 + 4 * (firstday_growth - 1))) / 2
        raw_values = np.array([1, r, r**2])
        growth_values = raw_values / raw_values.sum() * firstday_growth
        modified_hdwi['hourly_growth'][0:3] = growth_values
        print('POST modified hdwi:', modified_hdwi)
        modified_hdwi['hourly_growth'][0:3] = growth_values
    return modified_hdwi
    




