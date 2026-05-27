import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# just makes the 1pctCO2, abruptCO2 conc files

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

data_in = pd.read_csv(f'{indir}/rcmip_phase3_concentrations_{rcmip_version}.csv')

scens = ['1pctCO2', '1pctCO2-4xext', '1pctCO2-cdr', 'abrupt-4xCO2', 'abrupt-2xCO2', 'abrupt-0p5xCO2']

#%%

for scen in scens:
        
    in_filtered = data_in[
        (data_in['Scenario'] == scen) &
        (data_in['Region'] == 'World') &
        (data_in['Variable'] == 'Atmospheric Concentrations|CO2')
    ]
        
    
    year_cols = [
        c for c in in_filtered.columns
        if c.isdigit() and not pd.isna(in_filtered.iloc[0][c])
    ]
    
    in_long = in_filtered.melt(
        id_vars='Variable',
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    
    in_long['Year'] = in_long['Year'].astype(int)
    
    in_df = (
        in_long
        .pivot(index='Year', columns='Variable', values='Value')
        .sort_index()
    )
        
    
    in_df = in_df.rename(columns={
        'Atmospheric Concentrations|CO2': 'CO2 Forcing.Atmos CO2 exogenous'
    })

    in_df.to_csv(f'{outdir}/{scen}.csv')
    
