import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import scipy.stats

# this collates the input parameters for the priors - so we take the ocean
# parameters (on ocean_samples) and the FaIR ones (on samples), and combine
# to the full set, on samples

# have to run FRIDA-Clim_spinup.stmx before this

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
spinup_samples = int(os.getenv("SPINUP_SAMPLES"))

# load FaIR parameters - simple as these are just  on samples
# this bit is the same as in calibrate-FRIDA-climate

csv_list = ['aerosol_cloud', 'aerosol_radiation', 'carbon_cycle', 
           'climate_response_ebm3', 'forcing_scaling', 'ozone']

df_in = pd.DataFrame()

for csv in csv_list:
    df_csv = pd.read_csv(f"../data/external/samples_for_priors/{csv}_{samples}.csv")
    df_in = pd.concat([df_in, df_csv], axis=1)
    
fair_data_dict = {}

fair_vars_to_frida = {
'c1':'Energy Balance Model.Heat Capacity of Land & Ocean Surface[1]',
'c2':'Energy Balance Model.Heat Capacity of Thermocline Ocean[1]',
'c3':'Energy Balance Model.Heat Capacity of Deep Ocean[1]',
'epsilon':'Energy Balance Model.Deep Ocean Heat Uptake Efficacy Factor[1]',
'kappa1':'Energy Balance Model.Heat Transfer Coefficient between Land & Ocean Surface and Space[1]',
'kappa2':'Energy Balance Model.Heat Transfer Coefficient between Surface and Thermocline Ocean[1]',
'kappa3':'Energy Balance Model.Heat Transfer Coefficient between Thermocline Ocean and Deep Ocean[1]',
'beta':'Aerosol Forcing.Scaling Aerosol Cloud Interactions Effective Radiative Forcing scaling factor[1]',
'shape Sulfur':'Aerosol Forcing.Logarithmic Aerosol Cloud Interactions Effective Radiative Forcing scaling factor[1]',
'ari Sulfur':'Aerosol Forcing.Effective Radiative Forcing from Aerosol Radiation Interactions per unit SO2 Emissions[1]',
'scale CH4':'CH4 Forcing.Calibration scaling of CH4 forcing[1]',
'scale N2O':'N2O Forcing.Calibration scaling of N2O forcing[1]',
'scale minorGHG':'Minor GHGs Forcing.Calibration scaling of Minor GHG forcing[1]',
'scale Stratospheric water vapour':'Stratospheric Water Vapour Forcing.Calibration scaling of Stratospheric H2O forcing[1]',
'scale Light absorbing particles on snow and ice':'BC on Snow Forcing.Calibration scaling of Black Carbon on Snow forcing[1]',
'scale Albedo':'Land Use Forcing.Calibration scaling of Albedo forcing[1]',
'scale Irrigation':'Land Use Forcing.Calibration scaling of Irrigation forcing[1]',
'scale Volcanic':'Natural Forcing.Calibration scaling of Volcano forcing[1]',
'scale CO2':'CO2 Forcing.Calibration scaling of CO2 forcing[1]',
'solar_amplitude':'Natural Forcing.Amplitude of Effective Radiative Forcing from Solar Output Variations[1]',
'solar_trend':'Natural Forcing.Linear trend in Effective Radiative Forcing from Solar Output Variations[1]',
'o3 CH4':'Ozone Forcing.Ozone forcing per unit CH4 concentration change[1]',
'o3 N2O':'Ozone Forcing.Ozone forcing per unit N2O concentration change[1]',
'o3 Equivalent effective stratospheric chlorine':'Ozone Forcing.Ozone forcing per unit Montreal gases equivalent effective stratospheric chlorine concentration change[1]',
'o3 CO':'Ozone Forcing.Ozone forcing per unit CO emissions change[1]',
'o3 VOC':'Ozone Forcing.Ozone forcing per unit VOC emissions change[1]',
'o3 NOx':'Ozone Forcing.Ozone forcing per unit NOx emissions change[1]',
        }

for var in fair_vars_to_frida.keys():
    fair_data_dict[fair_vars_to_frida[var]] = df_in[var]

df_fair_data = pd.DataFrame(data=fair_data_dict, columns=fair_data_dict.keys())

#%%

# for FRIDA-Clim, add the land variables
land_variables = {
    "Forest.CO2 tree net primary production parameter[1]":[0.0005, 0.001],
    "Forest.STA maximum aboveground biomass per area parameter[1]":[0.2, 0.35],
    "Forest.STA squared maximum aboveground biomass per area parameter[1]":[-0.08, -0.02],
    "Forest.STA squared tree net primary production parameter[1]":[-0.05, -0.01],
    "Forest.STA tree net primary production parameter[1]":[0.01, 0.13],
    "Grass.CO2 grass net primary production parameter[1]":[0.0005, 0.001],
    "Grass.STA grass net primary production parameter[1]":[0.01, 0.13],
    "Grass.STA squared grass net primary production parameter[1]":[-0.05, -0.01],
    "Crop.harvest index for energy crops[1]":[0.6, 0.95],
    }

land_param_dict = {}

run_list = []
for i in np.arange(samples):
    run_list.append(f'Run {i+1}')
land_param_dict['Run'] = run_list

for l_i, land_var in enumerate(land_variables):
    
    land_param_dict[land_var] = scipy.stats.uniform.rvs(
        land_variables[land_var][0],
        land_variables[land_var][1] - land_variables[land_var][0],
        size=samples,
        random_state=3729329 + 1000*l_i,
    )
    
df_land = pd.DataFrame(land_param_dict, columns=land_param_dict.keys())


df_params = pd.concat([df_fair_data, df_land], axis=1)

#%%



# ocean temperature-linked parameters; again taken from calibrate-FRIDA-climate

ocean_variables = {
    "Ocean.Warm surface ocean alkalinity sensitivity to global T anomaly[1]":[-4e-6, -1e-6],
    "Ocean.Cold surface ocean alkalinity sensitivity to global T anomaly[1]":[-3e-5, -5e-6],
    "Ocean.High latitude carbon pump sensitivity to global T anomaly[1]":[-0.5,0],
    "Ocean.Warm surface ocean temperature sensitivity to global T anomaly[1]":[0.4, 1.0],
    "Ocean.Cold surface ocean temperature sensitivity to global T anomaly[1]":[0.3, 1.0],
    # "Ocean.Warm surface ocean salinity sensitivity to global T anomaly[1]":[-0.04, -0.01],
    # "Ocean.Cold surface ocean salinity sensitivity to global T anomaly[1]":[-0.3, -0.05],
    }


param_dict = {}

for o_i, ocean_var in enumerate(ocean_variables):
    
    param_dict[ocean_var] = scipy.stats.uniform.rvs(
        ocean_variables[ocean_var][0],
        ocean_variables[ocean_var][1] - ocean_variables[ocean_var][0],
        size=samples,
        random_state=3729329 + 1000*o_i,
    )
    
param_dict["Ocean.Warm surface ocean salinity sensitivity to global T anomaly[1]"
       ] = 10000*param_dict[
   "Ocean.Warm surface ocean alkalinity sensitivity to global T anomaly[1]"]

param_dict["Ocean.Cold surface ocean salinity sensitivity to global T anomaly[1]"
       ] = 10000*param_dict[
   "Ocean.Cold surface ocean alkalinity sensitivity to global T anomaly[1]"]

           
df_ocean_priors = pd.DataFrame(param_dict, columns=param_dict.keys())


df_params = pd.concat([df_params, df_ocean_priors], axis=1)


#%%

# process stocks from spinup

df_spinup_output = pd.read_csv(f'../data/spinup_output/Spinup_output_{spinup_samples}.csv')

variable_stock_list = [
    "Ocean.Cold surface ocean carbon reservoir[1]",
    "Ocean.Cold surface ocean pH[1]",
    "Ocean.Deep ocean ocean carbon reservoir[1]",
    "Ocean.Intermediate depth ocean carbon reservoir[1]",
    "Ocean.Warm surface ocean carbon reservoir[1]",
    "Ocean.Warm surface ocean pH[1]",
    "Forest.Mature forest aboveground biomass[1]",
    "Forest.Young forest aboveground biomass[1]",
    "Land Use.Mature Forest[1]",
    "Land Use.Young Forest[1]",
    "cropland soil carbon.fast soil carbon cropland[1]",
    "cropland soil carbon.slow soil carbon cropland[1]",
    "degraded land soil carbon.fast soil carbon degraded land[1]",
    "degraded land soil carbon.slow soil carbon degraded land[1]",
    "forest soil carbon.fast soil carbon mature forest[1]",
    "forest soil carbon.fast soil carbon young forest[1]",
    "forest soil carbon.slow soil carbon mature forest[1]",
    "forest soil carbon.slow soil carbon young forest[1]",
    "grassland soil carbon.fast soil carbon grassland[1]",
    "grassland soil carbon.slow soil carbon grassland[1]",
                    ]

stocks_dict = {}

variable_stock_list_frida = []
for variable_stock in variable_stock_list:
    variable_stock_list_frida.append(variable_stock.split(".")[0
                       ] + '.Initial ' + variable_stock.split(".")[1])

df_out = pd.DataFrame(columns=[variable_stock_list_frida])

for var in variable_stock_list:
    var_init = var.split(".")[0] + '.Initial ' + var.split(".")[1]
    
    stocks_dict[var_init] = np.full((spinup_samples), np.nan)
    for n_i in np.arange(spinup_samples):
        data_in = df_spinup_output[f'="Run {n_i+1}: {var}"'].values[0]
        stocks_dict[var_init][n_i] = data_in

df_spinup_stocks = pd.DataFrame(data=stocks_dict, columns=stocks_dict.keys())

# bring in spinup parameters
df_spinup_params = pd.read_csv(f"../data/spinup_input/spinup_params_{spinup_samples}.csv")

df_spinup = pd.concat([df_spinup_stocks, df_spinup_params], axis=1)

df_spinup= df_spinup.drop(columns='Forest.Young mature forest biomass ratio[1]')

# apply test(s) to make sure equilibrium is reached in the spinup
df_spinup_tests = pd.read_csv(f'../data/spinup_output/Spinup_output_tests_{spinup_samples}.csv')

idxs = np.full(spinup_samples, np.nan)

for i in np.arange(spinup_samples):
    if np.mean(np.abs(df_spinup_tests[f'="Run {i+1}: Ocean.Air sea co2 flux[1]"'])) < 0.01:
        idxs[i] = i

idxs = idxs[~np.isnan(idxs)]
n_kept = idxs.shape[0]
n_repeats = int(np.ceil(samples/n_kept))

# filter by idx 
df_spinup_filtered = df_spinup.iloc[idxs]

# repeat along index as needed, then crop down to match samples
df_spinup_out = pd.concat([df_spinup_filtered]*n_repeats, ignore_index=True)
df_spinup_out = df_spinup_out.iloc[:samples]


#%%
# finally combine parameters and stocks 

df_combined = pd.concat([df_params, df_spinup_out], axis=1)

df_run = df_combined["Run"].iloc[:,0]
df_combined = df_combined.drop(columns='Run')
df_combined = pd.concat([df_run, df_combined], axis=1)

df_combined = df_combined.rename(columns={
    "Ocean.Atmospheric CO2 Concentration 1750[1]": "CO2 Forcing.Atmospheric CO2 Concentration 1750[1]",
    })

df_combined.to_csv(
    f"../data/priors_input/priors_inputs_{samples}.csv",
    index=False,
)

