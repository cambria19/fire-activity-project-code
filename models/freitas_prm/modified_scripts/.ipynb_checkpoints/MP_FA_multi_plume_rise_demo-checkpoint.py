import pandas as pd 
import csv
import numpy as np
import multiprocessing as mp
import pandas as pd
from datetime import datetime, timedelta
import contextlib
import os
import subprocess, tempfile
from io import StringIO
import scipy.interpolate as interp
import tempdir_plume_rise_functions as prf
import shutil

###Function to be called each time a worker process is intialized 
#creates a new temporary directory for that worker process
def worker_init():
    global worker_tmp
    worker_tmp = tempfile.mkdtemp(prefix=f"worker_{os.getpid()}_")


###Function to process each row of the input CSV file, pulls required inputs and executes plume rise model
def process_row(inputs):
    
    fire_ID = str(inputs[4])
    date_time = inputs[0]
    burn_area = float(inputs[7]) #Burn area in m2
    fire_year = int(inputs[8])
    burn_area_ac = burn_area/4046.86
    lat = float(inputs[2])
    lon = float(inputs[3])
    heat_flux = float(inputs[6]) #Fire heat flux in kW/m2    
    
    plume_ID = f"{fire_ID}_{date_time}"   #Name of simulation, which is used to name model output. Should a string.
    wind_flag = 1                         #Flag determining if we want to turn on wind shear effects on the plume rise model. 1 = on, 0 = off       
    entrain   = 0.01                      #The entrainment coefficient. Generally leave this as .05.             
    fuel_moist = 10                       #Leave as is, not current used in the plume rise calculation in its current formulation    
            
    vertical_profile = np.arange(50,20000,100)         #Define heights for detrainment profile. Can be set to None if the
                                                       #user wants to stick with the default height profile
    #Path where output files will be placed.
    ROOT = Path.cwd().parents[2]
    output_path = ROOT / "model_outputs" / "pr_out"
    output_directory = output_path 
    
    write_plume_evo = False                             #Do we want to write out the evolution of the plume rise by timestep?
                                                        #(True/False). If not, discard the results. File can be larger.
    create_image = True                                 #Do we want to create a plot?
    output_filepath = f"{output_directory}/{plume_ID}"  #for later to add to output csv file

    #Merge namelist options together into a list of values
    plume_namelist = [heat_flux,burn_area,wind_flag,entrain,fuel_moist]

    #Where do we want to get our thermodynamic data from?
    vertprofile_path = ROOT / "data" / "creek_fire_vertical_profiles"
    thermo_file = f"{vertprofile_path}env_met_input_{fire_ID}_{date_time}.txt"

    ### Since we have a couple of empty grib files, skip files that are invalid for now
    if not os.path.exists(thermo_file):
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

    #Format the data accordinly where we need columns for "HGHT","PRES","TEMP","RELH","DWPT","DRCT","WIND","THTA","MIXR".
    #Units are as follows: "HGHT" = mASL, "PRES" = hPa, "TEMP" = C, "RELH" = %, "DWPT" = C, "DRCT" = degrees, "WIND" = m/s
    #"THTA" = K, and MIXR = g/kg. This is where the user will need to make the biggest changes, potentially. The final order 
    #needs to be as follows: ["HGHT","PRES","TEMP","RELH","DWPT","DRCT","WIND","THTA","MIXR"]
    thermo_profile = pd.read_fwf(thermo_file, skiprows=1,header=None)
    
    column_names = ["HGHT","PRES","TEMP","RELH","DWPT","DRCT","WIND","THTA","MIXR"]
    thermo_profile.columns = column_names
    thermo_profile["WIND"] = thermo_profile["WIND"]*0.514444     #Convert wind from kts to m/s

    #create temporary directory within broader parent directory (named based on worker process)
    tempdir = tempfile.mkdtemp(dir=worker_tmp)

    try:
        #Call the setup_plume_sim function. This will set up and run the plume rise model based on options selected for the
        #plume namelist options, the prescribed thermodynamic profile, and plume ID.
        plume_result, detrain_profile, pth = prf.setup_plume_sim(plume_namelist,plume_ID,thermo_profile,tempdir)  
  
        #compute_smoke_profile call moved inside setup_plume_sim
        
        #Call subroutine 'clean_up_run', which moves the output file into an output directory specified by the user. Naming of 
        #the output files determined by plume_ID
        prf.clean_up_run(plume_ID,output_directory,detrain_profile,write_plume_evo, tempdir)

    finally:
        #After we're done with function calls, delete the temporary directory created
        shutil.rmtree(tempdir) 

    pth_km = pth/1000 # convert pth from m to km

    detrain_profile['height(mAGL)'] = [x /1000 for x in detrain_profile['height(mAGL)']]

    output_list = [date_time, lat, lon, fire_ID, burn_area_ac, heat_flux, pth_km, output_filepath]

    return output_list  # return list of original model input and output from PRM to create output dataframe


### Parallelize
def process_row_parallel(row):
    result = process_row(row)
    return result
