## Overview
This repository contains the code used to develop the long-term wildfire activity dataset for California as part of the results and findings from: 
Drivers of Intensifying Wildfire Activity and Taller Wildfire Plumes Across California: Past Trends and Future Projections
The code for the modeling framework outlined in the study is provided for a sample wildfire event, the 2020 Creek Fire. The Creek Fire occurred during California’s record-breaking 2020 fire season and also resulted in the development of pyrocumulonimbus. As described in the study, the modeling framework comprises the First Order Fire Effects Model (FOFEM) (Reinhardt et al., 1997) and a plume rise model developed in (Freitas et al., 2007, 2010). The creation of model inputs, initialization of model runs, and the processing of model outputs is included for each model used. Visualizations of key model outputs are also included.

## FOFEM
The FOFEM model requires the inputs of meteorological conditions (windspeed and temperature), fuel moisture (1hr-1000hr) (Farguell et al., 2024), fuel classification (LANDFIRE, 2023), and wildfire location (Andela et al., 2019). Outputs include heat fluxes, emissions of key species, and the amount of fuel consumed. The FOFEM source code and installation instructions can be found in this repository: https://github.com/bran-jnw/fofem_wuinity . A translated version of the FOFEM model from C++ to Python will soon be made available. Additional scripts used to initialize FOFEM runs and process outputs are available in this repository.

## Plume Rise Model
The plume rise model requires inputs of active burning area, fire heat fluxes from FOFEM, and a thermodynamic profile. The plume rise model outputs estimated plume rise height, plume characteristics, and a detrainment profile. The plume rise model code and installation instructions can be found in this repository: https://github.com/tartanrunner25/Plume-model_stand_alone . Additional scripts used to initialize plume rise model runs and process outputs are available in this repository.

## Creek Fire Sample Case
Because of the scale of this long-term analysis, with hundreds of fires per year in some cases, the modeling framework is only demonstrated for the 2020 Creek Fire. The creation of the long-term analysis utilizes the same input/output processing and functions.
