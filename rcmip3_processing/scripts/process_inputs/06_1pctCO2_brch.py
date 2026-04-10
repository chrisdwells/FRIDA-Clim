import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# process initial stocks for the 1pctCO2 brch experiments

levels = ['750', '1000', '2000']

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
calibration = os.getenv("CALIBRATION")

df_posterior_params = pd.read_csv(
    f"../../data/constraining/frida_clim_inputs_{output_ensemble_size}_from_{samples}_1750_inits.csv",
)

df_1pct = pd.read_csv(
    "../../data/frida_clim_output/process_1pctCO2_for_brch.csv",
)
def load_frida(var):
    def process(df):
        var_out = np.full((len(df['Year']), output_ensemble_size), np.nan)
        for i in range(output_ensemble_size):
            colname = f'="Run {i+1}: {var}[1]"'
            var_out[:,i] = df[colname]
        return var_out
    return process


variable_stock_list = []
for row in df_1pct.columns:
    if "Run 1:" in row:
        varname = row.split(": ")[1].split("[1]")[0]
        variable_stock_list.append(varname)

constant_vars = ['CH4 Forcing.CH4 in atmosphere',
                 'Sea Level.Sea level anomaly from LWS',
                 'Minor GHGs Forcing.HFC134a eq in atmosphere', 
                 'N2O Forcing.Cumulative N2O emissions', 
                 'N2O Forcing.N2O in atmosphere',
                 'Terrestrial Carbon Balance.Cumulative terrestrial carbon balance',                 
                 'Terrestrial Carbon Balance.Peatland carbon balance',
                 'Cropland Carbon Balance.Cumulative cropland carbon balance',
                 'Forest carbon balance.Cumulative Forest carbon balance',
                 'Grassland carbon balance.Cumulative grassland carbon balance',
                 'Land Use.Cropland', 
                 'Land Use.Degraded Land', 
                 'Land Use.Grassland',
                 'CO2 Forcing.implied cumulative emissions',
                 
                 ]


variable_stock_list = [
    v for v in variable_stock_list
    if v not in constant_vars]


variable_stock_list_frida = []
for variable_stock in variable_stock_list:
    if variable_stock != 'CO2 Forcing.implied atmos CO2 anomaly':
        variable_stock_list_frida.append(variable_stock.split(".")[0
                           ] + '.Initial ' + variable_stock.split(".")[1] + '[1]')

df_posterior_params = df_posterior_params.drop(
    columns=variable_stock_list_frida,
    errors="ignore"
)

co2_ems = load_frida("CO2 Forcing.implied cumulative emissions")(df_1pct)

idxs = {}

for level in levels:
    expt = f'esm-1pct-brch-{level}PgC'
    idxs[expt] = (co2_ems > float(level)).argmax(axis=0)

    df_variable_inits_out = pd.DataFrame(columns=variable_stock_list_frida)
    
    for n_i in np.arange(output_ensemble_size):
        row = []
        
        for stock in variable_stock_list:
            # we don't want the normal atmos CO2 variable because this is the coupled response
            # term (ie negative); we need to use the enforced 1pctCO2 atmos change.
            if stock == 'CO2 Forcing.Atmospheric CO2 mass anomaly since 1750':
                row.append(df_1pct[f'="Run {n_i+1}: CO2 Forcing.implied atmos CO2 anomaly[1]"'].values[idxs[expt][n_i]])
            elif stock == 'CO2 Forcing.implied atmos CO2 anomaly':
                continue
            
            else:
                row.append(df_1pct[f'="Run {n_i+1}: {stock}[1]"'].values[idxs[expt][n_i]])
        
        df_variable_inits_out.loc[n_i] = row
        
    
    df_combined = pd.concat([df_posterior_params, df_variable_inits_out], axis=1)
    
    
    # need to set the normal mature forest biomass
    
    df_combined['Forest.Normal mature forest aboveground biomass per area[1]'
                        ] = df_combined['Forest.Initial Mature forest aboveground biomass[1]'
                        ]/df_combined['Land Use.Initial Mature Forest[1]']
    
    df_combined_cols = list(df_combined.keys())
    
    df_combined_newcols_df = pd.DataFrame(df_combined.values, columns=df_combined_cols)
    df_combined_newcols_df = df_combined_newcols_df.drop(['Run', 'Crop.crop yield 1980 reference[1]', 'Forest.forest aboveground biomass 1750[1]'], axis=1)
    
 
    
    df_combined_newcols_df.to_csv(
        f"../../data/processed_for_frida/esm_1pct_brch_{level}PgC_inputs_{output_ensemble_size}_from_{samples}_inits_params.csv",
        index=False,
    )
    
