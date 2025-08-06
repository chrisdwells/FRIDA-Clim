import os
import pandas as pd
import scipy.stats
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Generates the input parameters for FRIDA-Clim_spinup.stmx

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


NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
co2_1750_conc = scipy.stats.norm.rvs(
    size=spinup_samples, loc=278.3, scale=2.9 / NINETY_TO_ONESIGMA, random_state=1067061
)

# separately due to non-uniform
df_co2 = pd.DataFrame({"Ocean.Atmospheric CO2 Concentration 1750[1]": co2_1750_conc})

df = pd.concat([df, df_co2], axis=1)

os.makedirs("../data/spinup_input/", exist_ok=True)
df.to_csv(
    f"../data/spinup_input/spinup_params_{spinup_samples}.csv",
    index=False,
)

os.makedirs("../data/spinup_output/", exist_ok=True)
os.makedirs("../data/priors_input/", exist_ok=True)
os.makedirs("../data/priors_output/", exist_ok=True)
os.makedirs("../data/constraining/", exist_ok=True)

needed_csvs = [
    f'../data/spinup_output/Spinup_output_{spinup_samples}.csv',
    f'../data/spinup_output/Spinup_output_tests_{spinup_samples}.csv',
    '../data/priors_output/priors_land.csv',
    '../data/priors_output/priors_aerosols.csv',
    '../data/priors_output/priors_CO2.csv',
    '../data/priors_output/priors_ocean_CO2_flux.csv',
    '../data/priors_output/priors_ocean_heat_content.csv',
    '../data/priors_output/priors_temperature.csv',
    '../data/priors_output/priors_SLR.csv',
    '../data/priors_output/priors_NPP.csv',
    ]

for csv in needed_csvs:
    if os.path.isfile(csv) == False:
        df_blank = pd.DataFrame(list())
        df_blank.to_csv(csv)

