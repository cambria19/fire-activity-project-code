import csv
import pandas as pd
import json
import copy
from datetime import datetime
import glob
import os
import logging
import multiprocessing as mp
from pathlib import Path
current_path = Path(__file__).resolve()
import py_fofem.utils as utils
from py_fofem.run_fofem import main
import py_fofem.utils as utils

logging.basicConfig( level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s' )

# Function to split fofem inputs into chunks for multiprocessing
def chunk_inputs(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Function to process a chunk of fofem inputs
def process_chunk(chunk, num_name_dict):
    output_totals_chunk = []
    sim_num = 0
    for chunk_row in chunk:
        lat = float(chunk_row[0])
        lon = float(chunk_row[1])
        date_time = (chunk_row[2])
        fuelbed_name = chunk_row[3]
        timestep = int(chunk_row[4])
        fmois1kh = float(chunk_row[5])
        fmois100h = float(chunk_row[6])
        fmois10h = float(chunk_row[7])
        fmois1h = float(chunk_row[8])
        fmoislit = float(chunk_row[9])
        fmoisduff = float(chunk_row[10])
        percent_crown_burn = float(chunk_row[11])
        fuelcat = chunk_row[12]
        dufftype = chunk_row[13]
        met_dat_path = False
        ambient_temp = float(chunk_row[15])
        windspeed = float(chunk_row[16])
        emission_factors = chunk_row[17]
        fire_type = chunk_row[18]
        con_ob = False
        
        #pre processing
        if met_dat_path == "false" or met_dat_path == "FALSE" or met_dat_path == False:
            metdat_array = utils.generate_met_data_array(windspeed, ambient_temp)
        else:
            metdat_array =  utils.csv_to_array(met_dat_path)

        
        #run the model
        all_consumption, all_emissions, season, region, sim_length_seconds = main(lat, lon, date_time, fuelbed_name, timestep, fmois1kh, fmois100h, fmois10h, fmois1h, fmoislit, fmoisduff, percent_crown_burn, fuelcat, dufftype, metdat_array, emission_factors, fire_type, con_ob)
        sim_num = sim_num + 1
        #post processing               
        consumed_flaming = 0
        consumed_smoldering = 0

        for time_index in range(len(all_consumption)):
            consumed_flaming = consumed_flaming + all_consumption[time_index][13]
            consumed_smoldering = consumed_smoldering + all_consumption[time_index][15] 
            
            
        pm25, pm10, co, co2, ch4, nox, so2, fi_max, fi_last = 0, 0, 0, 0, 0, 0, 0, 0, 0
        
        burning = True
        num_timesteps = min(int((sim_length_seconds - 60)/15) + 1, 3000)
        for time_index in range(num_timesteps):
            duff_fi = all_emissions[time_index][16]  # fi = fire intensity
            crown_fi = all_emissions[time_index][17]
            fi_cur = all_emissions[time_index][15] + duff_fi # Dead woody + herb + shrub + duff + crown fire intensity
            
            #output flaming/smoldering emissions at each timestep
            pm25_fla   = all_emissions[time_index][1] 
            pm25_smo   = all_emissions[time_index][8]
            pm10_fla   = all_emissions[time_index][5] 
            pm10_smo   = all_emissions[time_index][12]
            co_fla     = all_emissions[time_index][3]
            co_smo     = all_emissions[time_index][10]
            co2_fla    = all_emissions[time_index][4] 
            co2_smo    = all_emissions[time_index][11] 
            ch4_smo    = all_emissions[time_index][9] 
            ch4_fla    = all_emissions[time_index][2] 
            nox_fla    = all_emissions[time_index][6] 
            nox_smo    = all_emissions[time_index][13]
            so2_fla    = all_emissions[time_index][7] 
            so2_smo    = all_emissions[time_index][14]
            # print('smoldering so2 emi:', so2_smo)

            model_output_list = [consumed_flaming, consumed_smoldering, pm25_fla, pm25_smo, pm10_fla, pm10_smo, co_fla, co_smo, co2_fla, co2_smo, ch4_fla, ch4_smo, nox_fla, nox_smo, so2_fla, so2_smo, fi_cur, crown_fi, time_index]
            chunk_row = list(chunk_row)
            new_row = chunk_row[:-3] + model_output_list + chunk_row[-3:]
            output_totals_chunk.append(new_row)


    return output_totals_chunk
    

def process_model_inputs(model_inputs, num_name_dict, chunk_size=None):
    num_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    if chunk_size is None:
        chunk_size = max(1, len(model_inputs) // (num_processes * 2))
        
    # Split model inputs into smaller chunks
    chunks = list(chunk_inputs(model_inputs, chunk_size))
    chunk_args = [(chunk, num_name_dict) for chunk in chunks]
    
    num_processes = min(num_processes, len(chunks))

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk_wrapper, chunk_args, chunksize=2)

    output_totals_array = [item for sublist in results for item in sublist]
    return output_totals_array

def process_chunk_wrapper(args):
    chunk, num_name_dict = args
    return process_chunk(chunk, num_name_dict)

# Write output arrays to CSV files
def write_output_to_csv(output_totals_array):
    # Path to output files 
    model_totals_path = os.path.join('/path/to/model/outputs/')  # Change this to where you would like to store your outputs
    
    # CSV headers
    totals_headers = ["lat", "lon", "date_time", "fuelbed_name", "timestep", "fmois1kh", "fmois100h", "fmois10h", "fmois1h", "fmoislit", "fmoisduff", "percent_crown_burn", "fuelcat", "dufftype", "met_dat_path", "ambient_temp", "windspeed", "emission_factors", "fire_type","con_ob", "consumed_flaming", "consumed_smoldering", "pm25_fla", "pm25_smo", "pm10_fla", "pm10_smo", "co_fla", "co_smo", "co2_fla", "co2_smo", "ch4_fla", "ch4_smo", "nox_fla", "nox_smo", "so2_fla", "so2_smo", "fi", "crown_fi",  "timeidx","fire_id","hourly_growth","FCCS_class_percent"]

    # Write output_totals_array to output CSV
    with open(model_totals_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(totals_headers)  # Write the header
        for row in output_totals_array:
            # Check if the first six columns are all zeroes
            if not all(value == 0 for value in row[22:34]):     # Check columns pm25 to fi (emisions and fire intensity) for when they are not all equal to zero
                writer.writerow(row)

    print(f"Output written to {model_totals_path}")

