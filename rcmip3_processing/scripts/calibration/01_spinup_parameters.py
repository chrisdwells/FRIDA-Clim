import os
import pandas as pd
import scipy.stats
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# RCMIP3 spinup inputs

spinup_samples = int(os.getenv("SPINUP_SAMPLES"))

spinup_variables = {
    # from climate module calibration
    "Ocean.Depth of warm surface ocean layer[1]":[50,500],
    "Ocean.Thickness of intermediate ocean layer[1]":[300,1000],
    "Ocean.Depth of cold surface ocean layer[1]":[50,500],
    "Ocean.Reference overturning strength in Sv[1]":[10,30],
    "Ocean.Reference intermediate to warm surface ocean mixing strength[1]":[50,90],
    "Ocean.Reference cold surface to deep ocean mixing strength[1]":[10,30],
    "Ocean.Reference strength of biological carbon pump in low latitude ocean[1]":[0,3],
    "Ocean.Reference strength of biological carbon pump in high latitude ocean[1]":[4,12],
    "Ocean.High latitude carbon pump transfer efficiency[1]":[0.1,0.5],
        
    # from sampleParmsParscaleRanged.csv - but ranges taken from wider model range
    "Forest.Young mature forest biomass ratio[1]":[0.3, 0.7],
    
    "Crop.normal harvest index for food crops[1]":[0.4, 0.45],
    "Crop.sensitivity of effect of crop yield on harvest index[1]":[0.05, 0.1],
    "Crop.crop yield 1980 reference[1]":[5.5, 6.5],
    "Crop.harvest index for feed crops[1]":[0.5	, 0.8],
    "Crop.sensitivity of effect of crop residue production on field fraction[1]":[0, 0.75],
    
    "soil carbon decay.cropland litter input share slow soil carbon[1]":[0.01, 0.03],
    "soil carbon decay.grassland litter input share slow soil carbon[1]":[0.01, 0.03],
    "soil carbon decay.mature forest litter input share slow soil carbon[1]":[0.015, 0.035],
    "soil carbon decay.young forest litter input share slow soil carbon[1]":[0.01, 0.035],
    
    "soil carbon decay.natural decay rate fast soil carbon[1]":[0.025, 0.035],
    "soil carbon decay.natural decay rate litter carbon[1]":[0.6, 0.8],
    "soil carbon decay.natural decay rate slow soil carbon[1]":[0.0008, 0.0012],
    "soil carbon decay.e0[1]":[290,300],
    "soil carbon decay.temp_response[1]":[55,60],
    
    "degraded land soil carbon.degraded land productivity reduction factor[1]":[0.01, 0.1],
    
    "Land Use.forest recovery time[1]":[50, 70],
    
    "Forest.tree net primary production in 1750[1]":[0.005, 0.0095],
    "Grass.grass net primary production in 1750[1]":[0.0035, 0.005],
    
    "Forest.forest aboveground biomass 1750[1]":[500, 700],
    
    "Land Use.Initial young forest area[1]":[50, 300],
    
    "Crop.historical sustainable farming fraction[1]" : [0.05,0.2],
    "Crop.mass to coverage[1]" : [889,1556],
    "Crop.conventional residue on field fraction[1]" : [0.3,0.45],
    "Crop.sustainable residue on field fraction[1]" : [0.65,0.85],
    "Terrestrial Carbon Balance.Forest carbon adaptation time[1]" : [10,50],

    }

param_dict = {}

run_list = []
for i in np.arange(spinup_samples):
    run_list.append(f'Run {i+1}')
param_dict['Run'] = run_list

for s_i, spinup_var in enumerate(spinup_variables):
    
    param_dict[spinup_var] = scipy.stats.uniform.rvs(
        spinup_variables[spinup_var][0],
        spinup_variables[spinup_var][1] - spinup_variables[spinup_var][0],
        size=spinup_samples,
        random_state=3729329 + 1000*s_i,
    )
    
df = pd.DataFrame(param_dict, columns=param_dict.keys())

os.makedirs("../../data/spinup_input/", exist_ok=True)
df.to_csv(
    f"../../data/spinup_input/spinup_params_{spinup_samples}.csv",
    index=False,
)

os.makedirs("../../data/spinup_output/", exist_ok=True)
os.makedirs("../../data/priors_input/", exist_ok=True)
os.makedirs("../../data/priors_output/", exist_ok=True)
os.makedirs("../../data/constraining/", exist_ok=True)
os.makedirs("../../data/posteriors_output/", exist_ok=True)

needed_csvs = [
    f'../../data/spinup_output/Spinup_output_{spinup_samples}.csv',
    f'../../data/spinup_output/Spinup_output_tests_{spinup_samples}.csv',
    '../../data/priors_output/priors_land.csv',
    '../../data/priors_output/priors_aerosols.csv',
    '../../data/priors_output/priors_aerosols_baseline.csv',
    '../../data/priors_output/priors_CO2.csv',
    '../../data/priors_output/priors_ocean_CO2_flux.csv',
    '../../data/priors_output/priors_ocean_heat_content.csv',
    '../../data/priors_output/priors_temperature.csv',
    '../../data/priors_output/priors_SLR.csv',
    '../../data/priors_output/priors_NPP_2000.csv',
    '../../data/priors_output/priors_NPP_2011.csv',
    
    '../../data/posteriors_output/posteriors_land.csv',
    '../../data/posteriors_output/posteriors_aerosols.csv',
    '../../data/posteriors_output/posteriors_CO2.csv',
    '../../data/posteriors_output/posteriors_ocean_CO2_flux.csv',
    '../../data/posteriors_output/posteriors_ocean_heat_content.csv',
    '../../data/posteriors_output/posteriors_temperature.csv',
    '../../data/posteriors_output/posteriors_SLR.csv',
    '../../data/posteriors_output/posteriors_NPP_2000.csv',
    '../../data/posteriors_output/posteriors_NPP_2011.csv',
    ]

for csv in needed_csvs:
    if os.path.isfile(csv) == False:
        df_blank = pd.DataFrame(list())
        df_blank.to_csv(csv)

