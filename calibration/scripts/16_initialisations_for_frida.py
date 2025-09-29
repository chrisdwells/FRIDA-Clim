import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# combine parameter sets and the 1980 stocks to generate the full set of 
# inputs to FRIDA IAM. This should only be done for the IAM calibration ie frida_iam

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
calibration = os.getenv("CALIBRATION")
calibration = "frida_iam"

# load parameter sets

df_posterior_params = pd.read_csv(
    f"../{calibration}/data/constraining/frida_clim_inputs_{output_ensemble_size}_from_{samples}_1750_inits.csv",
)

# load posterior 1980 stocks

df_1980_inits = pd.read_csv(
    f"../{calibration}/data/posteriors_output/posteriors_1980_stocks.csv",
)

# and 1993 ones for SLR offset

df_1993_inits = pd.read_csv(
    f"../{calibration}/data/posteriors_output/posteriors_1993_stocks.csv",
)
# and GMST which we need for the 1980 value of the 1850-1900 offset

df_temperature = pd.read_csv(
    f"../{calibration}/data/posteriors_output/posteriors_temperature.csv",
)

#%%

variable_stock_list = [
    "Energy Balance Model.Deep Ocean Temperature",
    "Energy Balance Model.Land & Ocean Surface Temperature",
    "Energy Balance Model.ocean heat content change",
    "Energy Balance Model.Thermocline Ocean Temperature",
    "CH4 Forcing.CH4 in atmosphere",
    "CO2 Forcing.Atmospheric CO2 mass anomaly since 1750",
    "Minor GHGs Forcing.HFC134a eq in atmosphere",
    "N2O Forcing.Cumulative N2O emissions",
    "N2O Forcing.N2O in atmosphere",
    "Ocean.Cold surface ocean carbon reservoir",
    "Ocean.Cold surface ocean pH",
    "Ocean.Deep ocean ocean carbon reservoir",
    "Ocean.Intermediate depth ocean carbon reservoir",
    "Ocean.Warm surface ocean carbon reservoir",
    "Ocean.Warm surface ocean pH",
    "Sea Level.AntIS Radius",
    "Sea Level.AntIS Volume",
    "Sea Level.Sea level anomaly from GrIS DIS",
    "Sea Level.Sea level anomaly from GrIS SMB",
    # "Sea Level.Sea level anomaly from LWS",
    "Sea Level.Sea level anomaly from mountain glaciers",
    "Sea Level.Sea level anomaly from thermal expansion",
    "Terrestrial Carbon Balance.Commited future soil carbon loss due to land-use transitions",
    # "Terrestrial Carbon Balance.Cumulative terrestrial carbon balance",
    # "Terrestrial Carbon Balance.Peatland carbon balance",
    # "Cropland Carbon Balance.Cumulative cropland carbon balance",
    # "Forest carbon balance.Cumulative Forest carbon balance",
    # "Grassland carbon balance.Cumulative grassland carbon balance",
    "Forest.Mature forest aboveground biomass",
    "Forest.Young forest aboveground biomass",
    "Land Use.Cropland",
    "Land Use.Degraded Land",
    "Land Use.Grassland",
    "Land Use.Mature Forest",
    "Land Use.Young Forest",
    "cropland soil carbon.fast soil carbon cropland",
    "cropland soil carbon.slow soil carbon cropland",
    "degraded land soil carbon.fast soil carbon degraded land",
    "degraded land soil carbon.slow soil carbon degraded land",
    "forest soil carbon.fast soil carbon mature forest",
    "forest soil carbon.fast soil carbon young forest",
    "forest soil carbon.slow soil carbon mature forest",
    "forest soil carbon.slow soil carbon young forest",
    "grassland soil carbon.fast soil carbon grassland",
    "grassland soil carbon.slow soil carbon grassland",
                    ]

offset_1993_stocks = [
    "Sea Level.Sea level anomaly from GrIS DIS",
    "Sea Level.Sea level anomaly from GrIS SMB",
    # "Sea Level.Sea level anomaly from LWS",
    "Sea Level.Sea level anomaly from mountain glaciers",
    "Sea Level.Sea level anomaly from thermal expansion",
    ]

constant_stocks = [
    "Land Use.Cropland",
    "Land Use.Degraded Land",
    "Land Use.Grassland",
    "Minor GHGs Forcing.HFC134a eq in atmosphere",
    "N2O Forcing.Cumulative N2O emissions",
    "N2O Forcing.N2O in atmosphere",
    ]

variable_stock_list_frida = []
for variable_stock in variable_stock_list:
    if variable_stock not in constant_stocks:
        variable_stock_list_frida.append(variable_stock.split(".")[0
                           ] + '.Initial ' + variable_stock.split(".")[1] + '[1]')

constant_stock_list_frida = []
for constant_stock in constant_stocks:
    constant_stock_list_frida.append(constant_stock.split(".")[0
                       ] + '.Initial ' + constant_stock.split(".")[1])

variable_column_list = variable_stock_list_frida + ['Energy Balance Model.Surface Temperature 1850 to 1900 offset relative to 1750[1]']

df_variable_inits_out = pd.DataFrame(columns=variable_column_list)
df_constant_inits_out = pd.DataFrame(columns=constant_stock_list_frida)

for n_i in np.arange(output_ensemble_size):
    row = []
    
    for stock in variable_stock_list:
        if stock not in constant_stocks:
            offset = 0
            if stock in offset_1993_stocks:
                offset = df_1993_inits[f'="Run {n_i+1}: {stock}[1]"'].values[0]
                
            row.append(df_1980_inits[f'="Run {n_i+1}: {stock}[1]"'].values[0] - offset)
    
    
    row.append(np.mean(df_temperature[f'="Run {n_i+1}: Energy Balance Model.Land & Ocean Surface Temperature[1]"'
                 ].loc[(df_temperature['Year'] >= 1850) & (df_temperature['Year'] <= 1900)].values))
    
    df_variable_inits_out.loc[n_i] = row
    
df_combined = pd.concat([df_posterior_params, df_variable_inits_out], axis=1)

df_combined_cols = list(df_combined.keys())

df_combined_newcols = [x.replace('[1]', '[*]') if isinstance(x, str) else x for x in df_combined_cols]
df_combined_newcols_df = pd.DataFrame(df_combined.values, columns=df_combined_newcols)
df_combined_newcols_df = df_combined_newcols_df.drop(['Run', 'Crop.crop yield 1980 reference[*]'], axis=1)

df_combined_newcols_df.to_csv(
    f"../{calibration}/data/constraining/frida_iam_inputs_{output_ensemble_size}_from_{samples}_1980_inits_params.csv",
    index=False,
)

row = []

for stock in constant_stocks:
        
    offset = 0
    if stock in offset_1993_stocks:
        offset = df_1993_inits[f'="Run 1: {stock}[1]"'].values[0]
        
    row.append(df_1980_inits[f'="Run 1: {stock}[1]"'].values[0] - offset)

df_constant_inits_out.loc[n_i] = row

df_constant_inits_out.to_csv(
    f"../{calibration}/data/constraining/frida_iam_inputs_{output_ensemble_size}_from_{samples}_1980_constant_stocks.csv",
    index=False,
)