import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Process outputs into format to go into FRIDA

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'

df_temp_obs = pd.read_csv(
    f'{indir}/rcmip_phase3_processed_constraining_data_{rcmip_version}.csv')



#%%
gmst_row = df_temp_obs.loc[
    df_temp_obs["Variable"] == "Global Mean Surface Temperature (GMST)"
        ]

years = [str(y) for y in range(1980, 2024)]
gmst = gmst_row[years].iloc[0].values
temp_posteriors = pd.read_csv('../../data/posteriors_output/posteriors_temperature.csv')

temp_in = temp_posteriors.loc[temp_posteriors['Year'] > 1979].values[:,1:
         ] - np.average(temp_posteriors.loc[(temp_posteriors['Year'
     ]>=1850) & (temp_posteriors['Year']<=1900)].drop(columns='Year').values, axis=0)

idxs_closest_to_obs = []

for n_i in np.arange(output_ensemble_size):
    rmse_in_obs = np.sqrt(((temp_in[:,n_i]-gmst)**2).mean())

    if n_i == 0:
        rmse_obs = rmse_in_obs
        idx_obs = n_i + 1 # in FRIDA they start from 1

    else:
        if rmse_in_obs < rmse_obs:
            rmse_obs = rmse_in_obs
            idx_obs = n_i + 1 # in FRIDA they start from 1
                
            
idxs_closest_to_obs.append(idx_obs)

#%%

plt.plot(years, temp_in, color='grey')
plt.plot(years, gmst, color='C0', label='Obs')
plt.plot(years, temp_in[:,idx_obs-1], color='C1', label=f'Median {idx_obs}')
plt.legend()

